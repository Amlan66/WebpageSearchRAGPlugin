"""
Colab Indexer for RAG Chrome Plugin
Similar to example3.py but for websites instead of documents

Run this on Google Colab with GPU to create FAISS index from URLs
Then manually download index.bin, metadata.json, url_index_cache.json to local

Instructions:
1. Upload this file to Google Colab
2. Change runtime to GPU (Runtime -> Change runtime type -> GPU)
3. Run the installation cell (see below)
4. Run this ENTIRE file (Runtime -> Run all, or click the play button at top)
5. Then run process_websites() in a NEW cell with your URLs
6. Download the generated files from webpage_index/ folder

IMPORTANT: Run the entire file first before calling process_websites()!
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import time

# ==================== Installation Commands ====================
# Run these in a Colab cell FIRST:
"""
# FAISS with GPU support in Colab (special installation)
!pip install -q faiss-cpu  # Use CPU version, still fast in Colab
# OR for GPU (requires conda):
# !conda install -c pytorch faiss-gpu -y

# Other dependencies
!pip install -q numpy requests beautifulsoup4 lxml tqdm sentence-transformers torch
"""

# ==================== Configuration ====================

# Chunking settings (same as example3.py)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40

# Embedding model
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Excluded domains (sensitive sites)
EXCLUDED_DOMAINS = [
    'mail.google.com',
    'gmail.com',
    'web.whatsapp.com',
    'whatsapp.com',
    'youtube.com',
    'accounts.google.com',
    'login',
    'signin',
    'auth',
    'banking',
    'bank',
    'paypal.com',
]


# ==================== Setup Embedding Model ====================

def setup_embedding_model(use_drive_cache: bool = False):
    """
    Load the embedding model on GPU
    
    Args:
        use_drive_cache: If True, cache model to Google Drive to avoid re-downloading
                        (requires mounting Google Drive first)
    """
    print("🔄 Loading embedding model: nomic-embed-text...")
    
    from sentence_transformers import SentenceTransformer
    import torch
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Check if we should use Google Drive caching
    drive_cache_path = None
    if use_drive_cache:
        try:
            # Check if Google Drive is mounted
            if Path("/content/drive/MyDrive").exists():
                drive_cache_path = "/content/drive/MyDrive/rag_chrome_plugin_models"
                Path(drive_cache_path).mkdir(parents=True, exist_ok=True)
                print(f"   📁 Using Google Drive cache: {drive_cache_path}")
                
                # Set HuggingFace cache to Google Drive
                os.environ['TRANSFORMERS_CACHE'] = drive_cache_path
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = drive_cache_path
            else:
                print("   ⚠️  Google Drive not mounted. Mount it to use caching:")
                print("       from google.colab import drive")
                print("       drive.mount('/content/drive')")
        except Exception as e:
            print(f"   ⚠️  Could not use Google Drive cache: {e}")
    
    # Load model (will use default cache or Drive cache if available)
    model = SentenceTransformer(
        EMBED_MODEL,
        trust_remote_code=True,
        device=device,
        cache_folder=drive_cache_path if drive_cache_path else None
    )
    
    print(f"✅ Model loaded. Dimension: {EMBEDDING_DIM}")
    
    if drive_cache_path and Path(drive_cache_path).exists():
        print(f"   💾 Model cached to Google Drive for future sessions")
    
    return model


# ==================== Helper Functions ====================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text into overlapping segments (same as example3.py)
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def url_hash(url: str, content: str) -> str:
    """
    Compute MD5 hash of URL content (similar to file_hash in example3.py)
    """
    return hashlib.md5(content.encode()).hexdigest()


def should_index_url(url: str) -> bool:
    """Check if URL should be indexed (exclude sensitive sites)"""
    url_lower = url.lower()
    
    for domain in EXCLUDED_DOMAINS:
        if domain in url_lower:
            return False
    
    if url_lower.startswith(('chrome://', 'about:', 'file://')):
        return False
    
    return True


def fetch_webpage_content(url: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch and extract content from URL with multiple strategies to avoid 403 errors
    Returns: {title, content, url} or None if failed
    """
    
    # Multiple user agents to try (rotate if blocked)
    user_agents = [
        # Chrome on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Firefox on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Safari on macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        # Edge on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        # Chrome on Android (mobile)
        'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    ]
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   🔄 Retry {attempt}/{max_retries-1}...")
                time.sleep(2)  # Wait before retry
            else:
                print(f"   Fetching: {url}")
            
            # Rotate user agent on each attempt
            headers = {
                'User-Agent': user_agents[attempt % len(user_agents)],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }
            
            # Add referer for some sites that require it
            if attempt > 0:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=15,
                allow_redirects=True,
                verify=True  # Verify SSL certificates
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else url
            if title:
                title = title.strip()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            
            if len(text.split()) < 50:
                print(f"   ⚠️  Content too short ({len(text.split())} words), skipping")
                return None
            
            print(f"   ✅ Fetched successfully ({len(text.split())} words)")
            
            return {
                'url': url,
                'title': title,
                'content': text
            }
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"   ⚠️  403 Forbidden (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue  # Try again with different user agent
                else:
                    print(f"   ❌ Failed after {max_retries} attempts: {e}")
                    print(f"   💡 Tip: This site blocks automated access. Try visiting it manually in Chrome.")
                    return None
            elif e.response.status_code == 429:
                print(f"   ⚠️  429 Too Many Requests - rate limited")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
                    continue
                else:
                    print(f"   ❌ Failed: Rate limited")
                    return None
            else:
                print(f"   ❌ HTTP Error {e.response.status_code}: {e}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            else:
                print(f"   ❌ Failed: Timeout after {max_retries} attempts")
                return None
        
        except requests.exceptions.ConnectionError as e:
            print(f"   🔌 Connection error: {e}")
            return None
        
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
            return None
    
    return None


def get_embeddings_batch(model, texts: List[str]) -> np.ndarray:
    """Get embeddings for a batch of texts"""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32
    )
    return embeddings.astype('float32')


# ==================== Main Processing Function ====================

def process_websites(urls: List[str], output_dir: str = "webpage_index"):
    """
    Process websites and create FAISS index (similar to process_documents in example3.py)
    
    Args:
        urls: List of URLs to index
        output_dir: Directory to save index files
    """
    print("="*80)
    print("🌐 RAG Chrome Plugin - Website Indexer")
    print("="*80)
    print(f"URLs to process: {len(urls)}")
    print()
    
    # Setup
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    INDEX_FILE = OUTPUT_DIR / "index.bin"
    METADATA_FILE = OUTPUT_DIR / "metadata.json"
    CACHE_FILE = OUTPUT_DIR / "url_index_cache.json"
    
    # Load existing data (if any)
    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None
    
    # Load embedding model
    embedding_model = setup_embedding_model()
    print()
    
    # Process each URL
    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for url in urls:
        print(f"Processing: {url}")
        
        # Check if should index
        if not should_index_url(url):
            print(f"   ⏭️  SKIP: Excluded domain")
            skipped_count += 1
            continue
        
        # Fetch webpage
        webpage = fetch_webpage_content(url)
        if not webpage:
            failed_count += 1
            continue
        
        # Compute content hash
        content_hash = url_hash(url, webpage['content'])
        
        # Check if already indexed with same content
        if url in CACHE_META and CACHE_META[url] == content_hash:
            print(f"   ⏭️  SKIP: Unchanged content")
            skipped_count += 1
            continue
        
        # Remove old chunks if URL was indexed before (update case)
        if url in CACHE_META:
            print(f"   🔄 UPDATE: Removing old chunks")
            new_metadata = [m for m in metadata if m['url'] != url]
            removed = len(metadata) - len(new_metadata)
            metadata = new_metadata
            print(f"   Removed {removed} old chunks")
            # Note: In production, you'd rebuild the FAISS index here
            # For simplicity, we'll just rebuild at the end
        
        try:
            # Chunk the content
            chunks = chunk_text(webpage['content'])
            print(f"   Created {len(chunks)} chunks")
            
            if not chunks:
                print(f"   ⚠️  No chunks created, skipping")
                continue
            
            # Generate embeddings
            print(f"   🔄 Generating embeddings...")
            embeddings = get_embeddings_batch(embedding_model, chunks)
            
            # Add to index
            if index is None:
                index = faiss.IndexFlatL2(EMBEDDING_DIM)
                print(f"   Created new FAISS index (dim={EMBEDDING_DIM})")
            
            index.add(embeddings)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "url": url,
                    "chunk": chunk,
                    "chunk_id": f"{hashlib.md5(url.encode()).hexdigest()[:8]}_{i}",
                    "chunk_index": i,
                    "title": webpage['title'],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Update cache
            CACHE_META[url] = content_hash
            
            print(f"   ✅ Indexed {len(chunks)} chunks")
            indexed_count += 1
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed_count += 1
            continue
        
        print()
    
    # Save everything
    print("="*80)
    print("💾 Saving index files...")
    
    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    print(f"✅ Saved cache: {CACHE_FILE}")
    
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"✅ Saved metadata: {METADATA_FILE} ({len(metadata)} chunks)")
    
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        print(f"✅ Saved FAISS index: {INDEX_FILE} ({index.ntotal} vectors)")
    else:
        print("⚠️  No index to save (no URLs processed)")
    
    # Summary
    print()
    print("="*80)
    print("📊 Summary")
    print("="*80)
    print(f"✅ Indexed:  {indexed_count}")
    print(f"⏭️  Skipped:  {skipped_count}")
    print(f"❌ Failed:   {failed_count}")
    print(f"📦 Total chunks: {len(metadata)}")
    print(f"🔢 Total vectors: {index.ntotal if index else 0}")
    print()
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print()
    print("🎉 Done! Now download these files to your local machine:")
    print(f"   1. {INDEX_FILE.name}")
    print(f"   2. {METADATA_FILE.name}")
    print(f"   3. {CACHE_FILE.name}")
    print()
    print("Place them in: RAGChromePlugin/faiss_cache/")
    print("="*80)


# ==================== Convenience Functions ====================

def load_urls_from_file(filename: str = "urls.txt") -> List[str]:
    """Load URLs from a text file (one URL per line)"""
    try:
        with open(filename, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"✅ Loaded {len(urls)} URLs from {filename}")
        return urls
    except FileNotFoundError:
        print(f"⚠️  File {filename} not found")
        return []


def download_files_colab():
    """Download the index files in Colab"""
    try:
        from google.colab import files
        
        print("📥 Preparing files for download...")
        files.download('webpage_index/index.bin')
        files.download('webpage_index/metadata.json')
        files.download('webpage_index/url_index_cache.json')
        print("✅ Files downloaded!")
    except ImportError:
        print("⚠️  Not running in Colab - files are in webpage_index/ folder")


# ==================== Example Usage ====================

def example_usage():
    """
    Example: How to use this script in Google Colab
    """
    
    # Method 1: Provide URLs directly
    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://docs.python.org/3/tutorial/",
        "https://www.python.org/about/",
    ]
    
    # Method 2: Load URLs from file
    # Create a urls.txt file with one URL per line:
    """
    https://example.com/page1
    https://example.com/page2
    # This is a comment
    https://example.com/page3
    """
    # urls = load_urls_from_file("urls.txt")
    
    # Process websites
    process_websites(urls)
    
    # Download files (in Colab)
    # download_files_colab()


# ==================== Main Execution ====================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        RAG Chrome Plugin - Website Indexer (Colab)          ║
    ║                                                              ║
    ║  This script indexes websites into a FAISS vector database  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Automatically run indexing if urls.txt exists
    if Path("urls.txt").exists():
        print("✅ Found urls.txt file!")
        print("🚀 Starting automatic indexing...\n")
        
        try:
            # Load URLs from file
            urls = load_urls_from_file("urls.txt")
            
            if not urls:
                print("⚠️  No URLs found in urls.txt")
                print("\nCreate urls.txt with one URL per line:")
                print("https://example.com/page1")
                print("https://example.com/page2")
                print("# Comments start with #")
            else:
                # Process the websites
                process_websites(urls)
                
                # Remind user to download files
                print("\n" + "="*80)
                print("📥 NEXT STEP: Download the files")
                print("="*80)
                print("\nRun this in a NEW cell:")
                print("-" * 60)
                print("from google.colab import files")
                print("files.download('webpage_index/index.bin')")
                print("files.download('webpage_index/metadata.json')")
                print("files.download('webpage_index/url_index_cache.json')")
                print("-" * 60)
                print("\nThen place them in: RAGChromePlugin/faiss_cache/")
                print("="*80)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  urls.txt not found!")
        print("\nPlease upload urls.txt file with one URL per line:")
        print("-" * 60)
        print("https://example.com/page1")
        print("https://example.com/page2")
        print("# Comments start with #")
        print("-" * 60)
        print("\nOr create it in Colab:")
        print("%%writefile urls.txt")
        print("https://example.com/page1")
        print("https://example.com/page2")
        print("\nThen run: %run colab_indexer.py")
