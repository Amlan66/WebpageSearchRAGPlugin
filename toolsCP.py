"""
Tools for RAG Chrome Plugin
Local tools for search, index management, and coordination with Colab
"""

import json
import requests
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from modelsCP import (
    SearchQuery, SearchResponse, SearchResult,
    IndexRequest, IndexResponse, IndexStatus,
    WebpageContent, WebpageChunk, ColabIndexRequest,
    ColabIndexResponse, IndexingProgress
)
from memoryCP import URLMemoryManager, IndexCacheManager, IndexOperationMemory
from perceptionCP import analyze_webpage_content, chunk_text
import datetime as dt


class LocalIndexManager:
    """
    READ-ONLY manager for local FAISS index
    
    This class ONLY reads the index files manually downloaded from Colab.
    It does NOT create or modify the index.
    
    Workflow:
    1. Index is created in Google Colab (colab_indexer.py)
    2. Files are manually downloaded to faiss_cache/
    3. This class loads and searches the index
    4. To update: re-run Colab, download new files, restart agent
    """
    
    def __init__(self, index_dir: str = "faiss_cache"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "index.bin"
        self.metadata_file = self.index_dir / "metadata.json"
        self.cache_file = self.index_dir / "url_index_cache.json"
        self.index_hash_file = self.index_dir / "index_hash.txt"  # MD5 hash of index
        
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict] = []
        
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                print(f"[IndexManager] Loaded index with {self.index.ntotal} vectors")
            
            if self.metadata_file.exists():
                self.metadata = json.loads(self.metadata_file.read_text())
                print(f"[IndexManager] Loaded {len(self.metadata)} metadata entries")
        except Exception as e:
            print(f"[IndexManager] Error loading index: {e}")
            self.index = None
            self.metadata = []
    
    def _save_index(self):
        """
        Save is disabled - index is managed via Colab only
        
        This method is kept for compatibility but does nothing.
        Index updates happen by re-running Colab and downloading fresh files.
        """
        print(f"[IndexManager] Save disabled - use Colab for index updates")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3, url_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Search the index with a query embedding
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            url_filter: Optional URL pattern to filter by
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or self.index.ntotal == 0:
            print("[IndexManager] Index is empty")
            return []
        
        try:
            # Search
            query_vec = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_vec, min(top_k * 3, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= len(self.metadata):
                    continue
                
                meta = self.metadata[idx]
                
                # Apply URL filter if specified
                if url_filter and url_filter not in meta["url"]:
                    continue
                
                # Convert distance to similarity score (inverse)
                similarity = 1.0 / (1.0 + float(dist))
                
                result = SearchResult(
                    url=meta["url"],
                    chunk_text=meta["chunk"],
                    chunk_id=meta["chunk_id"],
                    title=meta["title"],
                    similarity_score=similarity,
                    timestamp=meta.get("timestamp", "")
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            print(f"[IndexManager] Found {len(results)} results")
            return results
        
        except Exception as e:
            print(f"[IndexManager] Search error: {e}")
            return []
    
    def get_stats(self) -> IndexStatus:
        """Get current index statistics"""
        total_chunks = self.index.ntotal if self.index else 0
        
        # Count unique URLs
        unique_urls = set(meta["url"] for meta in self.metadata)
        
        # Get index file size
        index_size = self.index_file.stat().st_size if self.index_file.exists() else 0
        
        return IndexStatus(
            total_chunks=total_chunks,
            total_urls=len(unique_urls),
            index_size_bytes=index_size,
            last_updated=datetime.now().isoformat(),
            colab_available=False  # Will be set by agent
        )
    


class EmbeddingService:
    """Service to get embeddings - can use local or Colab"""
    
    def __init__(self, local_url: str = "http://localhost:11434/api/embeddings", model: str = "nomic-embed-text"):
        self.local_url = local_url
        self.model = model
        self.use_local = True  # Try local first
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.use_local:
            try:
                return self._get_local_embedding(text)
            except Exception as e:
                print(f"[Embedding] Local failed: {e}, trying Colab...")
                self.use_local = False
        
        # Fall back to Colab (would need implementation)
        return self._get_colab_embedding(text)
    
    def _get_local_embedding(self, text: str) -> np.ndarray:
        """Get embedding from local Ollama"""
        response = requests.post(
            self.local_url,
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)
    
    def _get_colab_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Colab (placeholder)"""
        # This would call Colab endpoint
        raise NotImplementedError("Colab embedding not yet implemented")


class ColabClient:
    """
    DISABLED - Colab client not used in manual indexing workflow
    
    This class is kept for compatibility but all methods are disabled.
    Indexing happens manually via colab_indexer.py script.
    """
    
    def __init__(self, colab_url: Optional[str] = None):
        self.colab_url = None
        self.available = False
        print("[Colab] Client disabled - manual indexing workflow only")


# ==================== Tool Functions ====================

async def search_index_tool(
    query: str,
    top_k: int = 3,
    url_filter: Optional[str] = None,
    index_manager: Optional[LocalIndexManager] = None,
    embedding_service: Optional[EmbeddingService] = None,
    colab_client: Optional[ColabClient] = None
) -> SearchResponse:
    """
    Search through indexed webpage content
    Uses manually downloaded index from Colab (no automatic sync)
    
    Args:
        query: Search query
        top_k: Number of results
        url_filter: Optional URL filter
        index_manager: Index manager instance
        embedding_service: Embedding service instance
        colab_client: Not used (kept for compatibility)
        
    Returns:
        SearchResponse with results
    """
    print(f"[Tool:Search] Query: '{query}', top_k: {top_k}")
    
    try:
        # Get query embedding (local Ollama for fast query embedding)
        query_embedding = embedding_service.get_embedding(query)
        
        # Search index (uses manually downloaded index from faiss_cache/)
        results = index_manager.search(query_embedding, top_k, url_filter)
        
        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results)
        )
    
    except Exception as e:
        print(f"[Tool:Search] Error: {e}")
        return SearchResponse(query=query, results=[], total_found=0)


async def index_webpage_tool(
    url: str,
    title: str,
    content: str,
    force_reindex: bool = False,
    index_manager: Optional[LocalIndexManager] = None,
    embedding_service: Optional[EmbeddingService] = None,
    url_memory: Optional[URLMemoryManager] = None,
    colab_client: Optional[ColabClient] = None
) -> IndexResponse:
    """
    Indexing is DISABLED - use manual Colab indexing instead
    
    To index new URLs:
    1. Add URLs to urls.txt
    2. Run colab_indexer.py in Google Colab
    3. Download index.bin, metadata.json, url_index_cache.json
    4. Place files in RAGChromePlugin/faiss_cache/
    5. Restart agent to reload index
    
    Args:
        url: URL to index
        title: Page title
        content: Page content
        force_reindex: Force reindexing
        index_manager: Index manager
        embedding_service: Embedding service
        url_memory: URL memory
        colab_client: Colab client
        
    Returns:
        IndexResponse with instructions
    """
    print(f"[Tool:Index] Indexing disabled - manual Colab indexing only")
    
    return IndexResponse(
        success=False,
        url=url,
        chunks_created=0,
        message=(
            "❌ Local indexing is disabled. "
            "Please use manual Colab indexing: "
            "1) Add URL to urls.txt in Colab, "
            "2) Run colab_indexer.py, "
            "3) Download and place index files in faiss_cache/, "
            "4) Restart agent."
        ),
        was_updated=False
    )


async def get_index_status_tool(
    index_manager: Optional[LocalIndexManager] = None,
    url_memory: Optional[URLMemoryManager] = None
) -> IndexStatus:
    """Get current index status"""
    stats = index_manager.get_stats()
    
    # Enhance with URL memory stats
    if url_memory:
        memory_stats = url_memory.get_stats()
        print(f"[Tool:Status] {memory_stats}")
    
    return stats


async def navigate_highlight_tool(
    url: str,
    chunk_text: str,
    chunk_id: str
) -> Dict[str, Any]:
    """
    Prepare navigation and highlight command for Chrome extension
    
    Args:
        url: URL to navigate to
        chunk_text: Text to highlight
        chunk_id: Chunk identifier
        
    Returns:
        Navigation command dictionary
    """
    print(f"[Tool:Navigate] URL: {url}, Chunk: {chunk_id}")
    
    return {
        "action": "navigate_and_highlight",
        "url": url,
        "text": chunk_text,
        "chunk_id": chunk_id,
        "timestamp": datetime.now().isoformat()
    }

