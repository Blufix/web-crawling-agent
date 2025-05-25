"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import time
import tiktoken

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents
import openai

# Initialize the tokenizer for token counting
ENCODING_MODEL = "cl100k_base"  # Model used by OpenAI for GPT-4 and text-embedding-3-*
tokenizer = tiktoken.get_encoding(ENCODING_MODEL)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the text using tiktoken.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Number of tokens in the text
    """
    encoded = tokenizer.encode(text)
    return len(encoded)

def estimate_embedding_tokens(texts: List[str]) -> int:
    """
    Estimate the number of tokens that would be used for embedding the given texts.
    
    Args:
        texts: List of texts to estimate token count for
        
    Returns:
        Estimated total number of tokens
    """
    return sum(count_tokens(text) for text in texts)

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Add to Supabase
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

async def safe_process_and_store_batch(
    supabase_client: Client,
    crawl_results: List[Dict[str, Any]],
    crawl_type: str,
    chunk_size: int,
    max_retries: int = 3,
) -> int:
    """Wrapper around process_and_store_batch with retry logic."""
    for attempt in range(max_retries):
        try:
            return await process_and_store_batch(
                supabase_client, crawl_results, crawl_type, chunk_size
            )
        except openai.RateLimitError as e:
            wait = 5 * (attempt + 1)
            print(
                f"Rate limit when storing batch. Waiting {wait}s before retry..."
            )
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise
            wait = 3 * (attempt + 1)
            print(f"Error storing batch: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls URLs sequentially to manage token usage
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links with token limit awareness
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    The crawling process stores results in batches, saving every 100 URLs or
    whenever the estimated tokens for the pending batch exceeds 200k to stay
    within OpenAI's token-per-minute limits.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)

    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Token usage tracking
        max_tpm = 200000  # Tokens per minute - 200k TPM limit
        max_tokens_per_batch = 40000  # Much higher batch limit since our TPM is 200k
        total_tokens_used = 0
        total_pages_crawled = 0
        all_crawl_results = []
        crawl_type = "webpage"
        batch_start_time = time.time()  # Track time for TPM calculations
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl - these are usually small so no need for batching
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
            all_crawl_results.extend(crawl_results)
            total_pages_crawled += len(crawl_results)
            
        elif is_sitemap(url):
            # For sitemaps, extract URLs but crawl them one by one or in small batches
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
                
            crawl_type = "sitemap"
            print(f"Found {len(sitemap_urls)} URLs in sitemap. Processing sequentially to manage token usage.")
            
            # Process sitemap URLs in small batches to avoid rate limit errors
            batch_size = 5  # Smaller batch size to prevent rate limit errors
            
            # Track accumulated results for periodic saving even if a batch is incomplete
            accumulated_results = []
            last_save_time = time.time()
            save_interval = 300  # Save at least every 5 minutes regardless of TPM limits
            
            for i in range(0, len(sitemap_urls), batch_size):
                batch_urls = sitemap_urls[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(sitemap_urls) + batch_size - 1)//batch_size}: {len(batch_urls)} URLs")
                
                # Crawl this small batch
                batch_results = await crawl_batch(crawler, batch_urls, max_concurrent=1)
                
                if batch_results:
                    # Add to accumulated results first
                    accumulated_results.extend(batch_results)
                    
                    # Process and store this batch immediately
                    batch_tokens = await safe_process_and_store_batch(
                        supabase_client, batch_results, crawl_type, chunk_size
                    )
                    
                    # Check token usage and apply rate limiting if needed
                    total_tokens_used += batch_tokens
                    total_pages_crawled += len(batch_results)
                    all_crawl_results.extend(batch_results)
                    
                    print(f"Batch complete. Tokens used: {batch_tokens}. Total tokens: {total_tokens_used}. Pages: {total_pages_crawled}")
                    
                    # Calculate current tokens per minute rate
                    elapsed_minutes = (time.time() - batch_start_time) / 60.0
                    elapsed_since_last_save = time.time() - last_save_time
                    
                    # Force save if it's been too long since the last save
                    if elapsed_since_last_save > save_interval:
                        print(f"Time-based save: {elapsed_since_last_save:.1f} seconds since last save.")
                        last_save_time = time.time()
                    
                    if elapsed_minutes > 0:
                        current_tpm = total_tokens_used / elapsed_minutes
                        print(f"Current TPM: {current_tpm:.2f} tokens/minute (Limit: {max_tpm})")
                        
                        # If we're approaching TPM limits, pause briefly
                        if current_tpm > max_tpm * 0.8:  # 80% of our TPM limit
                            # Save any accumulated data before pausing
                            if accumulated_results:
                                print(f"Saving {len(accumulated_results)} accumulated results before pausing")
                                # Note: We already saved this batch, so no need to save again
                                accumulated_results = []
                            
                            pause_time = 5  # Brief pause to regulate TPM
                            print(f"Approaching TPM limit ({current_tpm:.2f}/{max_tpm}). Pausing for {pause_time} seconds...")
                            await asyncio.sleep(pause_time)
                            # Reset timer after pause
                            batch_start_time = time.time()
                            last_save_time = time.time()
                            total_tokens_used = 0
        else:
            # For regular webpages, perform recursive crawling with token limits
            start_urls = [url]
            crawl_type = "webpage"
            
            # Crawl with token limit awareness - modify the function call to pass the supabase client
            # This allows the crawl_recursive_internal_links_with_token_limit function to
            # save results periodically rather than only at the end
            results = await crawl_recursive_internal_links_with_token_limit(
                crawler, 
                start_urls, 
                max_depth=max_depth, 
                max_concurrent=max_concurrent,
                max_tokens=max_tokens_per_batch,
                max_batch_size=30,
                max_tpm=max_tpm,
                supabase_client=supabase_client,  # Pass supabase client for periodic saving
                chunk_size=chunk_size,           # Pass chunk size for processing
                crawl_type=crawl_type,           # Pass crawl type for proper categorization
                max_urls_per_save=100,           # Save after 100 URLs
                max_tokens_per_save=max_tpm      # Respect overall token limit
            )
            
            if results:
                all_crawl_results.extend(results)
                total_pages_crawled += len(results)
                
                # The final batch may still need to be saved if not already saved in the function
                # We'll check if this final batch has already been saved inside the function
                if hasattr(results, 'needs_final_save') and results.needs_final_save:
                    total_tokens_used += await safe_process_and_store_batch(
                        supabase_client, results, crawl_type, chunk_size
                    )
        
        if not all_crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": total_pages_crawled,
            "total_tokens_used": total_tokens_used,
            "urls_crawled": [doc['url'] for doc in all_crawl_results][:5] + (["..."] if len(all_crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

async def process_and_store_batch(supabase_client: Client, crawl_results: List[Dict[str, Any]], 
                          crawl_type: str, chunk_size: int) -> int:
    """
    Process a batch of crawl results and store them in Supabase.
    Processes one document at a time with a 2-second delay between submissions to prevent rate limiting.
    Returns the estimated token usage.
    
    Args:
        supabase_client: Supabase client
        crawl_results: List of crawl results to process
        crawl_type: Type of crawl (sitemap, webpage, text_file)
        chunk_size: Maximum size of each content chunk
        
    Returns:
        Estimated token usage for this batch
    """
    from utils import add_single_document_to_supabase  # Import at function level to avoid circular imports
    
    chunk_count = 0
    all_contents = []  # Store all contents for token estimation
    
    print(f"Processing {len(crawl_results)} documents one at a time with a 2-second delay between submissions")
    
    for doc in crawl_results:
        source_url = doc['url']
        md = doc['markdown']
        chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
        
        print(f"Processing document: {source_url} - {len(chunks)} chunks")
        
        # Create full document mapping for this document
        url_to_full_document = {source_url: md}
        
        # Process each chunk of this document one at a time
        for i, chunk in enumerate(chunks):
            # Extract metadata
            meta = extract_section_info(chunk)
            meta["chunk_index"] = i
            meta["url"] = source_url
            meta["source"] = urlparse(source_url).netloc
            meta["crawl_type"] = crawl_type
            meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
            
            # Add content to the list for token estimation
            all_contents.append(chunk)
            
            # Process and store a single chunk
            print(f"Processing chunk {i+1}/{len(chunks)} for {source_url}")
            try:
                # Add single document with its full document context
                await add_single_document_to_supabase(
                    supabase_client, 
                    source_url, 
                    i, 
                    chunk, 
                    meta, 
                    url_to_full_document
                )
                
                # Wait 2 seconds between each submission to avoid rate limiting
                print(f"Waiting 2 seconds before next submission...")
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error processing chunk {i} of {source_url}: {e}")
                # Continue with the next chunk even if one fails
                await asyncio.sleep(3)  # Longer pause after an error
            
            chunk_count += 1
    
    # Estimate token usage for this batch
    estimated_tokens = estimate_embedding_tokens(all_contents)
    print(f"Processed {chunk_count} chunks with estimated {estimated_tokens} tokens")
    
    return estimated_tokens

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Crawl URLs one at a time to prevent rate limit errors.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions (no longer used as we process one by one)
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    
    # Process one URL at a time instead of in parallel
    results = []
    for url in urls:
        try:
            print(f"Processing single URL: {url}")
            result = await crawler.arun(url=url, config=crawl_config)
            
            if result.success and result.markdown:
                results.append({'url': result.url, 'markdown': result.markdown})
                print(f"Successfully processed {url}. Content length: {len(result.markdown)}")
                
                # Add a delay between requests to avoid rate limiting
                await asyncio.sleep(1)
            else:
                print(f"Failed to process {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
            # Continue with the next URL even if one fails
            await asyncio.sleep(2)  # Longer delay after an error
            continue
            
    return results

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls
        
    return results_all

async def crawl_recursive_internal_links_with_token_limit(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    max_tokens: int = 1800,
    max_batch_size: int = 10,
    max_tpm: int = 200000,
    supabase_client = None,  # Added parameter for Supabase client
    chunk_size: int = 5000,  # Added parameter for chunk size
    crawl_type: str = "webpage",  # Added parameter for crawl type
    max_urls_per_save: int = 100,  # New parameter controlling save batch size
    max_tokens_per_save: int = 200000  # New parameter controlling save token limit
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links with token limit awareness.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        max_tokens: Maximum tokens to process in a single batch
        max_batch_size: Maximum number of URLs to process in a batch
        max_tpm: Maximum tokens per minute rate limit
        supabase_client: Supabase client for periodic saving
        chunk_size: Maximum size of each content chunk
        crawl_type: Type of crawl (webpage, sitemap, text_file)
        max_urls_per_save: Save to Supabase after this many URLs are crawled
        max_tokens_per_save: Save to Supabase if estimated tokens exceed this value
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()
    to_visit_queue = [(url, 0) for url in start_urls]  # (url, depth)
    results_all = []
    current_token_estimate = 0
    batch_start_time = time.time()  # Initialize batch start time
    last_save_time = time.time()  # Track when we last saved to Supabase
    save_interval = 300  # Save at least every 5 minutes regardless of TPM limits
    pending_save_batch = []  # Track results that need to be saved
    already_saved_urls = set()  # Track URLs already saved to prevent duplicates
    tokens_since_save = 0  # Track estimated tokens since last save
    url_count_since_save = 0  # Track number of URLs since last save

    def normalize_url(url):
        return urldefrag(url)[0]
    
    # Process URLs in batches, respecting token limits
    while to_visit_queue:
        # Get a batch of URLs to process, respecting max depth and not exceeding max_batch_size
        current_batch = []
        next_queue = []
        
        for url_info in to_visit_queue:
            url, depth = url_info
            norm_url = normalize_url(url)
            
            if depth <= max_depth and norm_url not in visited and len(current_batch) < max_batch_size:
                current_batch.append(norm_url)
                visited.add(norm_url)  # Mark as visited to avoid duplicate processing
            else:
                next_queue.append(url_info)  # Put back in queue for next batch
                
        to_visit_queue = next_queue
        
        if not current_batch:
            break
            
        print(f"Processing batch of {len(current_batch)} URLs at max depth {max_depth}")
        
        # Crawl the current batch
        results = await crawler.arun_many(urls=current_batch, config=run_config, dispatcher=dispatcher)
        
        # Process results and update token count
        batch_content = []
        new_urls_to_visit = []
        
        for result in results:
            if result.success and result.markdown:
                # Estimate tokens for this content
                content_tokens = count_tokens(result.markdown)
                
                # If adding this content would exceed our token limit, process what we have and reset
                if current_token_estimate + content_tokens > max_tokens and batch_content:
                    print(f"Token limit approaching ({current_token_estimate}/{max_tokens}). Processing current batch.")
                    # Reset token count for next batch
                    current_token_estimate = 0
                    # Wait to respect rate limits
                    await asyncio.sleep(1)  
                
                # Add this result
                result_data = {'url': result.url, 'markdown': result.markdown}
                results_all.append(result_data)
                batch_content.append(result.markdown)
                current_token_estimate += content_tokens
                
                # Add to pending save batch if we have Supabase client
                if supabase_client and result.url not in already_saved_urls:
                    pending_save_batch.append(result_data)
                    tokens_since_save += content_tokens
                    url_count_since_save += 1
                
                # Collect internal links for next depth level
                current_depth = next((d for u, d in to_visit_queue if normalize_url(u) == normalize_url(result.url)), 0)
                if current_depth < max_depth:
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited and (next_url, current_depth + 1) not in to_visit_queue:
                            new_urls_to_visit.append((next_url, current_depth + 1))
        
        # Add new URLs to the visit queue
        to_visit_queue.extend(new_urls_to_visit)
        
        # Periodically save to Supabase if we have a client
        elapsed_since_last_save = time.time() - last_save_time
        time_to_save = elapsed_since_last_save > save_interval
        
        # Calculate tokens per minute for this batch
        elapsed_seconds = time.time() - batch_start_time
        if elapsed_seconds > 0:
            current_tpm = (current_token_estimate / elapsed_seconds) * 60
            print(f"Current TPM: {current_tpm:.2f} tokens/minute (Limit: {max_tpm})")

            save_due_to_size = (
                len(pending_save_batch) >= max_urls_per_save
                or tokens_since_save >= max_tokens_per_save
            )

            # If we need to save data (time, TPM, or batch size/token limit)
            if supabase_client and pending_save_batch and (
                time_to_save or current_tpm > max_tpm * 0.7 or save_due_to_size
            ):
                batch_size = len(pending_save_batch)
                print(f"Saving {batch_size} crawled pages to Supabase...")
                tokens_used = await safe_process_and_store_batch(
                    supabase_client, pending_save_batch, crawl_type, chunk_size
                )
                print(f"Saved {batch_size} pages, tokens used: {tokens_used}")
                for item in pending_save_batch:
                    already_saved_urls.add(item['url'])
                pending_save_batch = []
                last_save_time = time.time()
                tokens_since_save = 0
                url_count_since_save = 0
            
            # If we're approaching TPM limits, apply brief pause
            if current_tpm > max_tpm * 0.8:  # 80% of our TPM limit
                pause_time = 2  # Shorter pause to regulate TPM
                print(f"Approaching TPM limit ({current_tpm:.2f}/{max_tpm}). Brief pause for {pause_time} seconds...")
                await asyncio.sleep(pause_time)
                # Update the start time for accurate TPM calculation
                batch_start_time = time.time()
                current_token_estimate = 0
    
    # Final save of any remaining data
    if supabase_client and pending_save_batch:
        print(f"Final save: saving {len(pending_save_batch)} remaining crawled pages to Supabase...")
        await safe_process_and_store_batch(
            supabase_client, pending_save_batch, crawl_type, chunk_size
        )
        tokens_since_save = 0
        url_count_since_save = 0
        # Mark that there's no need for a final save in the caller
        results = []  # Create a new list to be able to add attributes
        results.extend(results_all)
        results.needs_final_save = False
        return results
    else:
        # If we don't have a Supabase client or no data is pending, the caller needs to save
        results = []  # Create a new list to be able to add attributes
        results.extend(results_all)
        results.needs_final_save = True
        return results
@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()
            
        # Use a set to efficiently track unique sources
        unique_sources = set()
        
        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())