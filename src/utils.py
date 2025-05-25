"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
import time
from typing import List, Dict, Any, Optional, Tuple
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
        
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Return empty embeddings if there's an error
        return [[0.0] * 1536 for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

# Global variables for tracking token usage in embedding generation
_embedding_last_reset_time = time.time()
_embedding_tokens_used = 0
_max_tpm = 150000  # 75% of the 200k TPM limit to be very conservative

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    Includes rate limiting to avoid OpenAI API rate limit errors.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    global _embedding_last_reset_time, _embedding_tokens_used, _max_tpm
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Skip contextual embedding if no model is set
    if not model_choice:
        return chunk, False
    
    # Check if we need to reset the token counter (1 minute has passed)
    current_time = time.time()
    elapsed_minutes = (current_time - _embedding_last_reset_time) / 60.0
    if elapsed_minutes >= 1.0:
        _embedding_tokens_used = 0
        _embedding_last_reset_time = current_time

    # Estimate token usage for this request
    # Input tokens: prompt + system message + full_document (truncated) + chunk
    # Use a more conservative token estimation - 4 tokens per word is more accurate for complex text
    estimated_input_tokens = (len(full_document[:25000].split()) + len(chunk.split()) + 100) * 4  # more conservative estimate
    # Output tokens: maximum response tokens
    estimated_output_tokens = 200 * 4  # max_tokens with conservative multiplier
    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
    
    # Check if we'd exceed the rate limit - use 70% threshold to be more cautious
    if _embedding_tokens_used + estimated_total_tokens > _max_tpm * 0.7:
        # Calculate how long to wait before trying again
        remaining_seconds = max(0, 60 - elapsed_minutes * 60)
        if remaining_seconds > 0:
            print(f"Rate limit approaching for contextual embedding. Waiting {remaining_seconds:.1f} seconds before trying again.")
            # In async context, you would use await asyncio.sleep(remaining_seconds)
            # For sync code, we'll use time.sleep
            time.sleep(min(remaining_seconds, 5))  # Cap at 5 seconds max wait
            # After waiting, recalculate elapsed time
            current_time = time.time()
            elapsed_minutes = (current_time - _embedding_last_reset_time) / 60.0
            if elapsed_minutes >= 1.0:
                _embedding_tokens_used = 0
                _embedding_last_reset_time = current_time
    
    # Maximum retry attempts for rate limit errors
    max_retries = 3
    retry_count = 0
    base_delay = 1.0  # Starting delay in seconds
    total_delay_time = 0  # Track cumulative delay time
    max_total_delay = 60  # Maximum total delay time in seconds before giving up
    
    while retry_count < max_retries and total_delay_time < max_total_delay:
        try:
            # Create the prompt for generating contextual information
            prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

            # Call the OpenAI API to generate contextual information
            response = openai.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Update token usage - add a safety buffer to be conservative
            actual_tokens = response.usage.total_tokens
            _embedding_tokens_used += actual_tokens * 1.2  # Add 20% buffer to be safe
            
            # Extract the generated context
            context = response.choices[0].message.content.strip()
            
            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"
            
            # Print token usage information
            print(f"Contextual embedding created. Tokens used: {actual_tokens}. Total TPM: {_embedding_tokens_used}/{_max_tpm}")
            
            return contextual_text, True
        
        except openai.RateLimitError as e:
            retry_count += 1
            # If too many retries or if TPM is severely exceeded, stop trying
            if retry_count >= max_retries or _embedding_tokens_used > _max_tpm * 1.5:
                print(f"Rate limit error after {retry_count} attempts or TPM exceeded by 50%. Using original chunk instead: {e}")
                print(f"Current TPM: {_embedding_tokens_used}/{_max_tpm}")
                return chunk, False
            
            # Extract wait time from error message if available, otherwise use exponential backoff
            wait_time = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
            if hasattr(e, 'response') and e.response and 'retry-after' in e.response.headers:
                wait_time = float(e.response.headers['retry-after'])
            elif 'Please try again in ' in str(e):
                # Try to extract wait time from error message
                try:
                    wait_str = str(e).split('Please try again in ')[1].split('ms')[0].strip()
                    if wait_str.endswith('ms'):
                        wait_time = float(wait_str[:-2]) / 1000  # Convert ms to seconds
                    elif wait_str.endswith('s'):
                        wait_time = float(wait_str[:-1])
                except Exception:
                    pass  # Use the default wait_time if extraction fails
            
            # Cap the wait time to avoid excessive waiting
            wait_time = min(wait_time, 15.0)  # Cap at 15 seconds max wait
            total_delay_time += wait_time
            
            # If total delay exceeds our threshold, give up
            if total_delay_time >= max_total_delay:
                print(f"Total delay time ({total_delay_time}s) exceeded max allowed ({max_total_delay}s). Using original chunk instead.")
                return chunk, False
                
            print(f"Rate limit hit, retrying in {wait_time:.2f} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
            
            # Reset token counter if we've waited long enough
            current_time = time.time()
            if (current_time - _embedding_last_reset_time) >= 60:
                _embedding_tokens_used = 0
                _embedding_last_reset_time = current_time
        
        except Exception as e:
            print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
            return chunk, False
    
    # If we exit the loop without returning, use the original chunk
    print("Maximum retries or delay time exceeded. Using original chunk.")
    return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

async def add_single_document_to_supabase(
    client: Client,
    source_url: str,
    chunk_number: int,
    content: str,
    metadata: Dict[str, Any],
    url_to_full_document: Dict[str, str]
) -> None:
    """
    Add a single document to Supabase with proper rate limiting and error handling.
    This function handles the entire process for one document chunk:
    1. Delete any existing documents with the same URL and chunk number
    2. Generate contextual embedding if enabled
    3. Create vector embedding
    4. Insert into Supabase
    
    Args:
        client: Supabase client
        source_url: URL of the document
        chunk_number: Chunk number within the document
        content: Content of the chunk
        metadata: Metadata for the chunk
        url_to_full_document: Mapping of URLs to full document content for context
    """
    # First, delete any existing document with the same URL and chunk number
    try:
        # Check if the URL and chunk_number combination exists
        result = client.table("crawled_pages").select("url").eq("url", source_url).eq("chunk_number", chunk_number).limit(1).execute()
        if result.data:
            # Delete the existing record
            print(f"Deleting existing record for {source_url}, chunk {chunk_number}")
            client.table("crawled_pages").delete().eq("url", source_url).eq("chunk_number", chunk_number).execute()
            # Brief pause after deletion
            await asyncio.sleep(0.5)
    except Exception as e:
        print(f"Error checking/deleting existing record: {e}")
        # Continue despite error - we'll try to insert anyway
    
    # Get the full document for context
    full_document = url_to_full_document.get(source_url, "")
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
    # Apply contextual embedding if enabled
    if use_contextual_embeddings and full_document:
        contextual_content, success = await generate_contextual_embedding_async(full_document, content)
        if success:
            metadata["contextual_embedding"] = True
    else:
        contextual_content = content
    
    # Create embedding with retries
    embedding = None
    max_embedding_retries = 3
    for attempt in range(max_embedding_retries):
        try:
            embedding = create_embedding(contextual_content)
            break
        except openai.RateLimitError as e:
            # Extract wait time from error message if available
            wait_time = 5.0 + (attempt * 3)  # Increasing wait time with each retry
            if 'Please try again in ' in str(e):
                try:
                    wait_str = str(e).split('Please try again in ')[1].split('s')[0].strip()
                    if wait_str.endswith('ms'):
                        wait_time = float(wait_str[:-2]) / 1000 + 2.0  # Add buffer
                    elif wait_str.endswith('s'):
                        wait_time = float(wait_str[:-1]) + 2.0  # Add buffer
                except Exception:
                    pass
            
            # Cap wait time and log the retry
            wait_time = min(wait_time, 15.0)
            print(f"Rate limit hit during embedding. Waiting {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_embedding_retries})")
            await asyncio.sleep(wait_time)
            
            # If last attempt, use a zero vector
            if attempt == max_embedding_retries - 1:
                print(f"Max embedding retries reached. Using placeholder embedding.")
                embedding = [0.0] * 1536  # Standard OpenAI embedding dimension
        except Exception as e:
            print(f"Error creating embedding: {e}")
            if attempt == max_embedding_retries - 1:
                print(f"Max embedding retries reached. Using placeholder embedding.")
                embedding = [0.0] * 1536  # Standard OpenAI embedding dimension
            else:
                await asyncio.sleep(2.0 * (attempt + 1))  # Wait with backoff
    
    # If embedding is still None, use a placeholder
    if embedding is None:
        embedding = [0.0] * 1536  # Standard OpenAI embedding dimension
    
    # Create the data to insert
    data = {
        "url": source_url,
        "chunk_number": chunk_number,
        "content": contextual_content,
        "metadata": {
            "chunk_size": len(contextual_content),
            **metadata
        },
        "embedding": embedding
    }
    
    # Insert into Supabase with retries
    max_insert_retries = 3
    for attempt in range(max_insert_retries):
        try:
            # Insert data into Supabase
            response = client.table("crawled_pages").insert(data).execute()
            print(f"Document {source_url}, chunk {chunk_number} inserted successfully")
            return  # Success, exit function
        except Exception as e:
            print(f"Error inserting document: {e}")
            if attempt < max_insert_retries - 1:
                # Wait with backoff before retry
                wait_time = 3.0 * (attempt + 1)
                print(f"Waiting {wait_time:.2f} seconds before retry... (Attempt {attempt+1}/{max_insert_retries})")
                await asyncio.sleep(wait_time)
            else:
                # Log the failed URL
                print(f"Failed to insert document after {max_insert_retries} attempts")
                with open("failed_urls.log", "a") as f:
                    f.write(f"{source_url},{chunk_number}\n")
                
                # Cache the data for later recovery
                cache_file = f"cache_{source_url.replace('/', '_').replace(':', '_')}_{chunk_number}.json"
                try:
                    with open(os.path.join("cache", cache_file), "w") as f:
                        json.dump(data, f)
                    print(f"Data cached to {cache_file} for later recovery")
                except Exception as cache_error:
                    print(f"Failed to cache data: {cache_error}")
                
                return  # Exit after max retries

async def generate_contextual_embedding_async(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Asynchronous version of generate_contextual_embedding to be used with async functions.
    
    Args:
        full_document: The full document text
        chunk: The chunk to generate embedding for
        
    Returns:
        Tuple of (contextual_text, success_boolean)
    """
    # Same logic as generate_contextual_embedding but with asyncio.sleep instead of time.sleep
    max_retries = 3
    retry_count = 0
    base_delay = 2.0
    max_total_delay = 30.0
    total_delay_time = 0.0
    model_choice = os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")
    
    global _embedding_tokens_used, _max_tpm, _embedding_last_reset_time
    
    # Check if we need to pause due to rate limits
    current_time = time.time()
    elapsed_minutes = (current_time - _embedding_last_reset_time) / 60.0
    
    # If we're close to the rate limit, pause
    if _embedding_tokens_used > _max_tpm * 0.8:
        remaining_seconds = max(0, 60 - elapsed_minutes * 60)
        if remaining_seconds > 0:
            print(f"Approaching token rate limit. Pausing for {remaining_seconds:.1f} seconds.")
            await asyncio.sleep(remaining_seconds)
            # Reset counter after waiting
            _embedding_tokens_used = 0
            _embedding_last_reset_time = time.time()
    
    while retry_count < max_retries and total_delay_time < max_total_delay:
        try:
            # Create the prompt for generating contextual information
            prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

            # Call the OpenAI API to generate contextual information
            response = openai.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Update token usage - add a safety buffer to be conservative
            actual_tokens = response.usage.total_tokens
            _embedding_tokens_used += actual_tokens * 1.2  # Add 20% buffer to be safe
            
            # Extract the generated context
            context = response.choices[0].message.content.strip()
            
            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"
            
            # Print token usage information
            print(f"Contextual embedding created. Tokens used: {actual_tokens}. Total TPM: {_embedding_tokens_used}/{_max_tpm}")
            
            return contextual_text, True
        
        except openai.RateLimitError as e:
            retry_count += 1
            # If too many retries or if TPM is severely exceeded, stop trying
            if retry_count >= max_retries or _embedding_tokens_used > _max_tpm * 1.5:
                print(f"Rate limit error after {retry_count} attempts or TPM exceeded by 50%. Using original chunk instead: {e}")
                print(f"Current TPM: {_embedding_tokens_used}/{_max_tpm}")
                return chunk, False
            
            # Extract wait time from error message if available, otherwise use exponential backoff
            wait_time = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
            if hasattr(e, 'response') and e.response and 'retry-after' in e.response.headers:
                wait_time = float(e.response.headers['retry-after'])
            elif 'Please try again in ' in str(e):
                # Try to extract wait time from error message
                try:
                    wait_str = str(e).split('Please try again in ')[1].split('ms')[0].strip()
                    if wait_str.endswith('ms'):
                        wait_time = float(wait_str[:-2]) / 1000  # Convert ms to seconds
                    elif wait_str.endswith('s'):
                        wait_time = float(wait_str[:-1])
                except Exception:
                    pass  # Use the default wait_time if extraction fails
            
            # Cap the wait time to avoid excessive waiting
            wait_time = min(wait_time, 15.0)  # Cap at 15 seconds max wait
            total_delay_time += wait_time
            
            # If total delay exceeds our threshold, give up
            if total_delay_time >= max_total_delay:
                print(f"Total delay time ({total_delay_time}s) exceeded max allowed ({max_total_delay}s). Using original chunk instead.")
                return chunk, False
                
            print(f"Rate limit hit, retrying in {wait_time:.2f} seconds... (Attempt {retry_count}/{max_retries})")
            await asyncio.sleep(wait_time)
            
            # Reset token counter if we've waited long enough
            current_time = time.time()
            if (current_time - _embedding_last_reset_time) >= 60:
                _embedding_tokens_used = 0
                _embedding_last_reset_time = current_time
        
        except Exception as e:
            print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
            return chunk, False
    
    # If we exit the loop without returning, use the original chunk
    print("Maximum retries or delay time exceeded. Using original chunk.")
    return chunk, False

def add_documents_to_supabase(
    client: Client, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> int:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Track existing URLs to avoid unnecessary deletions
    existing_urls_to_delete = []
    
    # Check if URLs already exist in the database before deleting
    # This helps minimize unnecessary deletions
    if unique_urls:
        try:
            # Check which URLs actually exist in the database
            for i in range(0, len(unique_urls), 20):  # Process in small batches
                batch = unique_urls[i:i+20]
                for url in batch:
                    try:
                        # Check if the URL exists before attempting to delete
                        result = client.table("crawled_pages").select("url").eq("url", url).limit(1).execute()
                        if result.data:  # Only add to delete list if it exists
                            existing_urls_to_delete.append(url)
                    except Exception:
                        # If we can't check, assume it might exist
                        existing_urls_to_delete.append(url)
                
                # Brief pause between batches to avoid overwhelming the database
                time.sleep(0.1)
        except Exception as e:
            print(f"Error checking existing URLs: {e}")
            # If we can't check, assume all URLs might exist
            existing_urls_to_delete = unique_urls
    
    # Only attempt deletion if we have URLs that exist
    if existing_urls_to_delete:
        print(f"Deleting {len(existing_urls_to_delete)} existing URLs from database")
        
        # Delete in smaller batches to avoid overwhelming the database
        delete_batch_size = 10
        for i in range(0, len(existing_urls_to_delete), delete_batch_size):
            batch = existing_urls_to_delete[i:i+delete_batch_size]
            try:
                # Use the .in_() filter to delete all records with matching URLs
                client.table("crawled_pages").delete().in_("url", batch).execute()
                print(f"Deleted batch {i//delete_batch_size + 1}/{(len(existing_urls_to_delete) + delete_batch_size - 1)//delete_batch_size}")
            except Exception as e:
                print(f"Batch delete failed: {e}. Trying one-by-one deletion.")
                # Fallback: delete records one by one
                for url in batch:
                    try:
                        client.table("crawled_pages").delete().eq("url", url).execute()
                    except Exception as inner_e:
                        print(f"Error deleting record for URL {url}: {inner_e}")
            
            # Brief pause between batches to avoid overwhelming the database
            time.sleep(0.2)
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Split into smaller batches for embedding to avoid rate limits
        embedding_sub_batch_size = 10
        all_embeddings = []
        
        for j in range(0, len(contextual_contents), embedding_sub_batch_size):
            sub_batch_end = min(j + embedding_sub_batch_size, len(contextual_contents))
            sub_batch = contextual_contents[j:sub_batch_end]
            
            # Check if we need to pause due to rate limits before creating embeddings
            global _embedding_last_reset_time, _embedding_tokens_used, _max_tpm
            current_time = time.time()
            elapsed_minutes = (current_time - _embedding_last_reset_time) / 60.0
            
            # Estimate tokens for this sub-batch - use a more conservative estimate
            estimated_tokens = sum(len(text.split()) for text in sub_batch) * 4  # more conservative token estimation
            
            # If we're close to the rate limit, pause - use 60% threshold to be very cautious
            if _embedding_tokens_used + estimated_tokens > _max_tpm * 0.6:
                remaining_seconds = max(0, 60 - elapsed_minutes * 60)
                if remaining_seconds > 0:
                    print(f"Approaching embedding rate limit. Pausing for {min(5, remaining_seconds):.1f} seconds.")
                    time.sleep(min(5, remaining_seconds))
                    # Reset counter if a minute has passed
                    current_time = time.time()
                    if (current_time - _embedding_last_reset_time) >= 60:
                        _embedding_tokens_used = 0
                        _embedding_last_reset_time = current_time
            
            # Create embeddings for this sub-batch with limited retries
            max_embedding_retries = 2
            embedding_retry_count = 0
            embedding_success = False
            
            while embedding_retry_count <= max_embedding_retries and not embedding_success:
                try:
                    # If we've already retried multiple times, use smaller batches
                    if embedding_retry_count > 0:
                        print(f"Retrying with smaller batch size after previous failure")
                        # Process one at a time on retries to minimize issues
                        mini_embeddings = []
                        for single_text in sub_batch:
                            try:
                                single_embedding = create_embedding(single_text)
                                mini_embeddings.append(single_embedding)
                                time.sleep(0.5)  # Brief pause between individual requests
                            except Exception as mini_e:
                                print(f"Error creating individual embedding: {mini_e}")
                                # Use a zero vector as placeholder for failed embeddings
                                mini_embeddings.append([0.0] * 1536)  # Standard OpenAI embedding dimension
                        
                        sub_batch_embeddings = mini_embeddings
                    else:
                        # Normal batch processing on first attempt
                        sub_batch_embeddings = create_embeddings_batch(sub_batch)
                    
                    all_embeddings.extend(sub_batch_embeddings)
                    
                    # Update token usage with a safety buffer
                    actual_tokens = estimated_tokens * 1.5  # Add 50% buffer to be extra safe
                    _embedding_tokens_used += actual_tokens
                    print(f"Embedded sub-batch {j//embedding_sub_batch_size + 1}/{(len(contextual_contents) + embedding_sub_batch_size - 1)//embedding_sub_batch_size}. Estimated tokens: {estimated_tokens}. Total TPM: {_embedding_tokens_used}/{_max_tpm}")
                    embedding_success = True
                    
                except openai.RateLimitError as e:
                    embedding_retry_count += 1
                    
                    # If we've reached max retries, use placeholder embeddings and continue
                    if embedding_retry_count > max_embedding_retries:
                        print(f"Max embedding retries reached after {max_embedding_retries} attempts. Using placeholder embeddings.")
                        # Use zero vectors as placeholders
                        placeholder_embeddings = [[0.0] * 1536 for _ in range(len(sub_batch))]  # Standard OpenAI embedding dimension
                        all_embeddings.extend(placeholder_embeddings)
                        break
                    
                    # Extract wait time from error message if available
                    wait_time = 5.0 + (embedding_retry_count * 3)  # Increasing wait time with each retry
                    if 'Please try again in ' in str(e):
                        try:
                            wait_str = str(e).split('Please try again in ')[1].split('ms')[0].strip()
                            if wait_str.endswith('ms'):
                                wait_time = float(wait_str[:-2]) / 1000 + 2.0  # Add buffer
                            elif wait_str.endswith('s'):
                                wait_time = float(wait_str[:-1]) + 2.0  # Add buffer
                        except Exception:
                            pass
                    
                    # Cap the wait time at 15 seconds
                    wait_time = min(wait_time, 15.0)
                    
                    # Force token counter reset if we hit a rate limit
                    _embedding_tokens_used = _max_tpm * 0.8  # Set to 80% of max to ensure we'll wait properly
                    
                    print(f"Rate limit hit during embedding. Waiting {wait_time:.2f} seconds... (Attempt {embedding_retry_count}/{max_embedding_retries})")
                    time.sleep(wait_time)
                    
                    # Reset token counter if we've waited long enough
                    current_time = time.time()
                    if (current_time - _embedding_last_reset_time) >= 60:
                        _embedding_tokens_used = 0
                        _embedding_last_reset_time = current_time
                
                except Exception as e:
                    embedding_retry_count += 1
                    print(f"Error creating embeddings: {e}")
                    
                    # If we've reached max retries, use placeholder embeddings and continue
                    if embedding_retry_count > max_embedding_retries:
                        print(f"Max embedding retries reached after {max_embedding_retries} attempts. Using placeholder embeddings.")
                        # Use zero vectors as placeholders
                        placeholder_embeddings = [[0.0] * 1536 for _ in range(len(sub_batch))]  # Standard OpenAI embedding dimension
                        all_embeddings.extend(placeholder_embeddings)
                        break
                    
                    # Wait before retry
                    wait_time = 2.0 * embedding_retry_count
                    print(f"Waiting {wait_time:.2f} seconds before retry... (Attempt {embedding_retry_count}/{max_embedding_retries})")
                    time.sleep(wait_time)
            
            # More substantial pause between sub-batches to avoid rate limiting
            if j + embedding_sub_batch_size < len(contextual_contents):
                time.sleep(2.0)  # Longer pause between batches
                
                # Check if we should reset our counter based on elapsed time
                current_time = time.time()
                if (current_time - _embedding_last_reset_time) >= 60:
                    print("One minute elapsed, resetting token counter")
                    _embedding_tokens_used = 0
                    _embedding_last_reset_time = current_time
        
        # Use the collected embeddings
        batch_embeddings = all_embeddings
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "embedding": batch_embeddings[j]  # Use embedding from contextual content
            }
            
            batch_data.append(data)
        
        # Insert batch into Supabase with limited retries
        max_insert_retries = 2
        insert_retry_count = 0
        insert_success = False
        
        while insert_retry_count <= max_insert_retries and not insert_success:
            try:
                # Insert data into Supabase
                client.table("crawled_pages").insert(batch_data).execute()
                print(f"Batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size} inserted successfully")
                insert_success = True
            
            except Exception as e:
                insert_retry_count += 1
                print(f"Error inserting batch: {e}")
                
                # If we've reached max retries, log and continue
                if insert_retry_count > max_insert_retries:
                    print(f"Max insert retries reached after {max_insert_retries} attempts. Continuing with next batch.")
                    # Log the URLs that failed to be inserted
                    with open("failed_urls.log", "a") as f:
                        for item in batch_data:
                            f.write(f"{item['url']},{item['chunk_number']}\n")
                    break
                
                # Wait before retry with increasing backoff
                wait_time = 3.0 * insert_retry_count
                print(f"Waiting {wait_time:.2f} seconds before retry... (Attempt {insert_retry_count}/{max_insert_retries})")
                time.sleep(wait_time)

def search_documents(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []