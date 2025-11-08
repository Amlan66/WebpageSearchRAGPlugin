"""
Memory management for RAG Chrome Plugin
Tracks visited URLs, indexing state, and search history
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from modelsCP import URLMemoryItem, IndexMemoryItem
import hashlib


class URLMemoryManager:
    """Manages memory of visited URLs and their indexing state"""
    
    def __init__(self, storage_path: str = "url_memory.json"):
        self.storage_path = Path(storage_path)
        self.url_memory: Dict[str, URLMemoryItem] = {}
        self._load()
    
    def _load(self):
        """Load URL memory from disk"""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self.url_memory = {
                    url: URLMemoryItem(**item) 
                    for url, item in data.items()
                }
            except Exception as e:
                print(f"[URLMemory] Failed to load: {e}")
                self.url_memory = {}
    
    def _save(self):
        """Save URL memory to disk"""
        try:
            data = {
                url: item.model_dump() 
                for url, item in self.url_memory.items()
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[URLMemory] Failed to save: {e}")
    
    def add_or_update_url(
        self, 
        url: str, 
        title: str, 
        content_hash: str,
        is_indexed: bool = False,
        chunk_count: int = 0
    ) -> URLMemoryItem:
        """Add new URL or update existing one"""
        now = datetime.now().isoformat()
        
        if url in self.url_memory:
            item = self.url_memory[url]
            item.last_visited = now
            item.visit_count += 1
            item.title = title
            item.content_hash = content_hash
            item.is_indexed = is_indexed
            item.chunk_count = chunk_count
        else:
            item = URLMemoryItem(
                url=url,
                title=title,
                first_visited=now,
                last_visited=now,
                visit_count=1,
                is_indexed=is_indexed,
                content_hash=content_hash,
                chunk_count=chunk_count
            )
            self.url_memory[url] = item
        
        self._save()
        return item
    
    def get_url(self, url: str) -> Optional[URLMemoryItem]:
        """Get memory item for a URL"""
        return self.url_memory.get(url)
    
    def is_url_indexed(self, url: str) -> bool:
        """Check if URL is indexed"""
        item = self.get_url(url)
        return item.is_indexed if item else False
    
    def get_content_hash(self, url: str) -> Optional[str]:
        """Get stored content hash for a URL"""
        item = self.get_url(url)
        return item.content_hash if item else None
    
    def has_content_changed(self, url: str, new_hash: str) -> bool:
        """Check if content has changed since last index"""
        old_hash = self.get_content_hash(url)
        if old_hash is None:
            return True  # New URL
        return old_hash != new_hash
    
    def mark_as_indexed(self, url: str, chunk_count: int):
        """Mark URL as indexed"""
        if url in self.url_memory:
            self.url_memory[url].is_indexed = True
            self.url_memory[url].chunk_count = chunk_count
            self._save()
    
    def get_all_indexed_urls(self) -> List[str]:
        """Get list of all indexed URLs"""
        return [
            url for url, item in self.url_memory.items() 
            if item.is_indexed
        ]
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        total_urls = len(self.url_memory)
        indexed_urls = len(self.get_all_indexed_urls())
        total_chunks = sum(
            item.chunk_count 
            for item in self.url_memory.values()
        )
        
        return {
            "total_urls_visited": total_urls,
            "indexed_urls": indexed_urls,
            "total_chunks": total_chunks,
            "unindexed_urls": total_urls - indexed_urls
        }


class IndexOperationMemory:
    """Tracks indexing operations history"""
    
    def __init__(self, storage_path: str = "index_operations.json"):
        self.storage_path = Path(storage_path)
        self.operations: List[IndexMemoryItem] = []
        self._load()
    
    def _load(self):
        """Load operation history from disk"""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self.operations = [
                    IndexMemoryItem(**item) 
                    for item in data
                ]
            except Exception as e:
                print(f"[IndexMemory] Failed to load: {e}")
                self.operations = []
    
    def _save(self):
        """Save operation history to disk"""
        try:
            data = [item.model_dump() for item in self.operations]
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[IndexMemory] Failed to save: {e}")
    
    def add_operation(
        self, 
        operation: str, 
        url: str, 
        chunks_affected: int
    ):
        """Record an indexing operation"""
        item = IndexMemoryItem(
            operation=operation,
            url=url,
            timestamp=datetime.now().isoformat(),
            chunks_affected=chunks_affected
        )
        self.operations.append(item)
        
        # Keep only last 1000 operations
        if len(self.operations) > 1000:
            self.operations = self.operations[-1000:]
        
        self._save()
    
    def get_recent_operations(self, limit: int = 10) -> List[IndexMemoryItem]:
        """Get recent operations"""
        return self.operations[-limit:]
    
    def get_url_history(self, url: str) -> List[IndexMemoryItem]:
        """Get operation history for a specific URL"""
        return [
            op for op in self.operations 
            if op.url == url
        ]


class IndexCacheManager:
    """Manages FAISS index cache (similar to doc_index_cache.json in example3.py)"""
    
    def __init__(self, cache_path: str = "faiss_cache/url_index_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}  # url -> content_hash
        self._load()
    
    def _load(self):
        """Load cache from disk"""
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception as e:
                print(f"[IndexCache] Failed to load: {e}")
                self.cache = {}
    
    def _save(self):
        """Save cache to disk"""
        try:
            self.cache_path.write_text(json.dumps(self.cache, indent=2))
        except Exception as e:
            print(f"[IndexCache] Failed to save: {e}")
    
    def compute_content_hash(self, content: str) -> str:
        """Compute MD5 hash of content (like example3.py)"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def should_index(self, url: str, content: str) -> bool:
        """Determine if URL should be indexed based on content hash"""
        content_hash = self.compute_content_hash(content)
        
        if url not in self.cache:
            return True  # New URL
        
        return self.cache[url] != content_hash  # Content changed
    
    def update_cache(self, url: str, content: str):
        """Update cache with new content hash"""
        content_hash = self.compute_content_hash(content)
        self.cache[url] = content_hash
        self._save()
    
    def get_cached_hash(self, url: str) -> Optional[str]:
        """Get cached hash for a URL"""
        return self.cache.get(url)
    
    def remove_url(self, url: str):
        """Remove URL from cache"""
        if url in self.cache:
            del self.cache[url]
            self._save()
    
    def get_all_cached_urls(self) -> List[str]:
        """Get all URLs in cache"""
        return list(self.cache.keys())


class SearchHistoryMemory:
    """Tracks user search queries and results"""
    
    def __init__(self, storage_path: str = "search_history.json", max_size: int = 500):
        self.storage_path = Path(storage_path)
        self.max_size = max_size
        self.history: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load search history from disk"""
        if self.storage_path.exists():
            try:
                self.history = json.loads(self.storage_path.read_text())
            except Exception as e:
                print(f"[SearchHistory] Failed to load: {e}")
                self.history = []
    
    def _save(self):
        """Save search history to disk"""
        try:
            self.storage_path.write_text(json.dumps(self.history, indent=2))
        except Exception as e:
            print(f"[SearchHistory] Failed to save: {e}")
    
    def add_search(self, query: str, results_count: int, top_url: Optional[str] = None):
        """Record a search query"""
        entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": results_count,
            "top_url": top_url
        }
        self.history.append(entry)
        
        # Keep only max_size entries
        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size:]
        
        self._save()
    
    def get_recent_searches(self, limit: int = 10) -> List[Dict]:
        """Get recent searches"""
        return self.history[-limit:]
    
    def get_popular_queries(self, limit: int = 10) -> List[tuple]:
        """Get most popular search queries"""
        query_counts = {}
        for entry in self.history:
            query = entry["query"]
            query_counts[query] = query_counts.get(query, 0) + 1
        
        sorted_queries = sorted(
            query_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_queries[:limit]

