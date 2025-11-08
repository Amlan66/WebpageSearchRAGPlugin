"""
Data models for RAG Chrome Plugin
Defines structures for webpage indexing, search, and navigation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ==================== Webpage Models ====================

class WebpageContent(BaseModel):
    """Raw webpage content captured by Chrome extension"""
    url: str
    title: str
    content: str  # HTML or extracted text
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebpageChunk(BaseModel):
    """A chunk of webpage content with metadata"""
    url: str
    chunk_text: str
    chunk_id: str
    chunk_index: int
    title: str
    timestamp: str


class WebpageHash(BaseModel):
    """Hash tracking for webpage content change detection"""
    url: str
    content_hash: str
    last_indexed: str
    chunk_count: int


# ==================== Indexing Models ====================

class IndexRequest(BaseModel):
    """Request to index a webpage"""
    url: str
    title: str
    content: str
    force_reindex: bool = False


class IndexResponse(BaseModel):
    """Response after indexing operation"""
    success: bool
    url: str
    chunks_created: int
    message: str
    was_updated: bool = False  # True if existing content was replaced


class IndexStatus(BaseModel):
    """Current status of the FAISS index"""
    total_chunks: int
    total_urls: int
    index_size_bytes: int
    last_updated: str
    colab_available: bool


# ==================== Search Models ====================

class SearchQuery(BaseModel):
    """Search query from Chrome extension"""
    query: str
    top_k: int = 3
    url_filter: Optional[str] = None  # Filter by specific URL


class SearchResult(BaseModel):
    """A single search result with context"""
    url: str
    chunk_text: str
    chunk_id: str
    title: str
    similarity_score: float
    timestamp: str


class SearchResponse(BaseModel):
    """Complete search response"""
    query: str
    results: List[SearchResult]
    total_found: int


# ==================== Navigation Models ====================

class HighlightRequest(BaseModel):
    """Request to highlight text on a webpage"""
    url: str
    text_to_highlight: str
    chunk_id: Optional[str] = None


class NavigationRequest(BaseModel):
    """Request to navigate to a URL and highlight text"""
    url: str
    chunk_text: str
    chunk_id: str


class NavigationResponse(BaseModel):
    """Response after navigation"""
    success: bool
    url: str
    message: str


# ==================== Agent Models ====================

class PerceptionResultCP(BaseModel):
    """Perception of user's search intent"""
    user_query: str
    intent: str  # 'search_content', 'check_index', 'navigate_to_result'
    entities: List[str] = []
    search_hint: Optional[str] = None


class ActionResultCP(BaseModel):
    """Result of an action executed by the agent"""
    action_name: str
    success: bool
    result: Any
    message: str


class DecisionCP(BaseModel):
    """Decision made by the agent"""
    action: str  # 'search_index', 'navigate_highlight', 'fetch_index', 'index_webpage'
    parameters: Dict[str, Any]
    reasoning: str


# ==================== Memory Models ====================

class URLMemoryItem(BaseModel):
    """Memory item tracking visited URLs"""
    url: str
    title: str
    first_visited: str
    last_visited: str
    visit_count: int
    is_indexed: bool
    content_hash: str
    chunk_count: int


class IndexMemoryItem(BaseModel):
    """Memory item for index operations"""
    operation: str  # 'indexed', 'updated', 'deleted'
    url: str
    timestamp: str
    chunks_affected: int


# ==================== Colab Communication Models ====================

class ColabIndexRequest(BaseModel):
    """Request sent to Colab for indexing"""
    webpage: WebpageContent
    existing_chunks_to_remove: List[str] = []  # Chunk IDs to remove if updating


class ColabIndexResponse(BaseModel):
    """Response from Colab after indexing"""
    success: bool
    url: str
    chunks_created: int
    embeddings_count: int
    error: Optional[str] = None


class ColabFetchRequest(BaseModel):
    """Request to fetch index from Colab"""
    fetch_type: str = "full"  # 'full' or 'incremental'
    last_sync_timestamp: Optional[str] = None


class ColabFetchResponse(BaseModel):
    """Response when fetching index from Colab"""
    success: bool
    index_data: Optional[bytes] = None  # FAISS index binary
    metadata: Optional[Dict[str, Any]] = None
    cache_data: Optional[Dict[str, str]] = None  # URL hash cache
    timestamp: str
    total_chunks: int


# ==================== Progress Tracking Models ====================

class IndexingProgress(BaseModel):
    """Progress of indexing operation"""
    url: str
    total_chunks: int
    processed_chunks: int
    percentage: float
    status: str  # 'chunking', 'embedding', 'indexing', 'complete', 'error'
    message: str


class ProgressUpdate(BaseModel):
    """Real-time progress update sent to Chrome extension"""
    operation_id: str
    progress: IndexingProgress
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

