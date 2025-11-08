"""
Agent for RAG Chrome Plugin
FastAPI server that orchestrates perception, decision, action, and memory
"""

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from datetime import datetime

# Import agentic components
from perceptionCP import extract_perception, analyze_webpage_content
from decisionCP import make_decision, make_indexing_decision
from actionCP import ActionExecutor
from memoryCP import URLMemoryManager, IndexCacheManager, IndexOperationMemory, SearchHistoryMemory
from modelsCP import (
    WebpageContent, SearchQuery, SearchResponse, IndexRequest,
    IndexResponse, IndexStatus, NavigationRequest, IndexingProgress
)
from toolsCP import (
    LocalIndexManager, EmbeddingService, ColabClient,
    search_index_tool, index_webpage_tool, get_index_status_tool,
    navigate_highlight_tool
)


# ==================== FastAPI App ====================

app = FastAPI(
    title="RAG Chrome Plugin Agent",
    description="Agentic backend for indexing and searching webpages",
    version="1.0.0"
)

# CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Chrome extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Global State ====================

class AgentState:
    """Global agent state"""
    def __init__(self):
        # Memory managers
        self.url_memory = URLMemoryManager()
        self.index_cache = IndexCacheManager()
        self.operation_memory = IndexOperationMemory()
        self.search_history = SearchHistoryMemory()
        
        # Index and services
        self.index_manager = LocalIndexManager()
        self.embedding_service = EmbeddingService()
        self.colab_client = ColabClient()
        
        # Tools registry
        self.tools = {
            "search_index": self._search_wrapper,
            "index_webpage": self._index_wrapper,
            "get_index_status": self._status_wrapper,
            "navigate_highlight": self._navigate_wrapper
        }
        
        # Action executor
        self.action_executor = ActionExecutor(self.tools)
        
        # WebSocket connections for progress updates
        self.websocket_connections: List[WebSocket] = []
    
    async def _search_wrapper(self, query: str, top_k: int = 3, url_filter: Optional[str] = None):
        """Wrapper for search tool"""
        return await search_index_tool(
            query=query,
            top_k=top_k,
            url_filter=url_filter,
            index_manager=self.index_manager,
            embedding_service=self.embedding_service,
            colab_client=None  # No Colab sync - manual index updates only
        )
    
    async def _index_wrapper(self, url: str, title: str, content: str, force_reindex: bool = False):
        """Wrapper for index tool"""
        return await index_webpage_tool(
            url=url,
            title=title,
            content=content,
            force_reindex=force_reindex,
            index_manager=self.index_manager,
            embedding_service=self.embedding_service,
            url_memory=self.url_memory,
            colab_client=self.colab_client
        )
    
    async def _status_wrapper(self):
        """Wrapper for status tool"""
        return await get_index_status_tool(
            index_manager=self.index_manager,
            url_memory=self.url_memory
        )
    
    async def _navigate_wrapper(self, url: str, chunk_text: str, chunk_id: str):
        """Wrapper for navigation tool"""
        return await navigate_highlight_tool(url, chunk_text, chunk_id)
    
    async def broadcast_progress(self, progress: IndexingProgress):
        """Broadcast progress to all connected WebSocket clients"""
        for ws in self.websocket_connections:
            try:
                await ws.send_json(progress.model_dump())
            except:
                pass


# Initialize global state
state = AgentState()


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "RAG Chrome Plugin Agent",
        "version": "1.0.0"
    }


@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Search through indexed webpage content
    
    This is the main search endpoint used by Chrome extension
    """
    try:
        # 1. Perception: Understand the query
        perception = extract_perception(query.query)
        print(f"[Agent] Perception: {perception.intent}")
        
        # 2. Decision: Decide what to do
        decision = make_decision(
            perception=perception,
            url_memory=state.url_memory,
            search_history=state.search_history.get_recent_searches()
        )
        print(f"[Agent] Decision: {decision.action}")
        
        # 3. Action: Execute search
        if decision.action == "search_index":
            result = await state.action_executor.execute(
                action_name="search_index",
                parameters=decision.parameters
            )
            
            if result.success:
                # Record search in history
                search_result = result.result
                top_url = search_result.results[0].url if search_result.results else None
                state.search_history.add_search(
                    query=query.query,
                    results_count=len(search_result.results),
                    top_url=top_url
                )
                
                return search_result
            else:
                raise HTTPException(status_code=500, detail=result.message)
        else:
            # If decision is not search, execute it and then search
            await state.action_executor.execute(
                action_name=decision.action,
                parameters=decision.parameters
            )
            # Fall back to search
            result = await search_index_tool(
                query=query.query,
                top_k=query.top_k,
                url_filter=query.url_filter,
                index_manager=state.index_manager,
                embedding_service=state.embedding_service
            )
            return result
    
    except Exception as e:
        print(f"[Agent] Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_webpage(request: IndexRequest):
    """
    ❌ INDEXING DISABLED - Manual Colab indexing only
    
    This endpoint is disabled because indexing happens manually via Google Colab.
    
    Workflow for adding new URLs:
    1. Add URLs to urls.txt in Google Colab
    2. Run: %run colab_indexer.py
    3. Download: index.bin, metadata.json, url_index_cache.json
    4. Place files in: RAGChromePlugin/faiss_cache/
    5. Restart this agent to reload the index
    """
    print(f"[Agent] Indexing request for: {request.url} - DISABLED")
    
    return IndexResponse(
        success=False,
        url=request.url,
        chunks_created=0,
        message=(
            "❌ Automatic indexing is disabled. "
            "Use manual Colab workflow: "
            "(1) Add URL to urls.txt in Colab → "
            "(2) Run colab_indexer.py → "
            "(3) Download index files → "
            "(4) Place in faiss_cache/ → "
            "(5) Restart agent"
        ),
        was_updated=False
    )


@app.get("/status", response_model=IndexStatus)
async def get_status():
    """Get current indexing status and statistics"""
    try:
        status = await get_index_status_tool(
            index_manager=state.index_manager,
            url_memory=state.url_memory
        )
        
        # Add Colab availability
        status.colab_available = state.colab_client.available
        
        return status
    
    except Exception as e:
        print(f"[Agent] Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/navigate")
async def navigate(request: NavigationRequest):
    """
    Navigate to a URL and highlight text
    
    Returns command for Chrome extension to execute
    """
    try:
        result = await navigate_highlight_tool(
            url=request.url,
            chunk_text=request.chunk_text,
            chunk_id=request.chunk_id
        )
        return result
    
    except Exception as e:
        print(f"[Agent] Navigation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_url")
async def check_url(url: str):
    """
    Check if URL is indexed and if content has changed
    
    Called by Chrome extension on page load
    """
    try:
        url_item = state.url_memory.get_url(url)
        
        if url_item:
            return {
                "indexed": url_item.is_indexed,
                "visit_count": url_item.visit_count,
                "last_visited": url_item.last_visited,
                "chunk_count": url_item.chunk_count
            }
        else:
            return {
                "indexed": False,
                "visit_count": 0,
                "last_visited": None,
                "chunk_count": 0
            }
    
    except Exception as e:
        print(f"[Agent] Check URL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ColabConfig(BaseModel):
    """Configuration for Colab connection"""
    ngrok_url: str


@app.post("/config/colab")
async def configure_colab(config: ColabConfig):
    """
    Configure Colab connection
    
    User provides ngrok URL from Colab notebook
    """
    try:
        state.colab_client.set_url(config.ngrok_url)
        return {
            "success": True,
            "message": f"Connected to Colab at {config.ngrok_url}"
        }
    
    except Exception as e:
        print(f"[Agent] Colab config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def get_memory_stats():
    """Get comprehensive memory statistics"""
    try:
        url_stats = state.url_memory.get_stats()
        recent_searches = state.search_history.get_recent_searches(5)
        recent_operations = state.operation_memory.get_recent_operations(5)
        popular_queries = state.search_history.get_popular_queries(5)
        
        return {
            "url_memory": url_stats,
            "recent_searches": recent_searches,
            "recent_operations": [op.model_dump() for op in recent_operations],
            "popular_queries": [{"query": q, "count": c} for q, c in popular_queries]
        }
    
    except Exception as e:
        print(f"[Agent] Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint for real-time indexing progress updates
    """
    await websocket.accept()
    state.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        state.websocket_connections.remove(websocket)
        print("[Agent] WebSocket client disconnected")


# ==================== Agentic Query Processing ====================

class AgenticQuery(BaseModel):
    """Natural language query to agent"""
    query: str
    context: Optional[Dict[str, Any]] = None


@app.post("/agent/query")
async def process_agentic_query(request: AgenticQuery):
    """
    Process natural language query through full agentic loop
    
    This demonstrates the full perception -> decision -> action cycle
    """
    try:
        print(f"[Agent] Processing query: '{request.query}'")
        
        # Step 1: Perception
        perception = extract_perception(request.query)
        print(f"[Agent] Perception: Intent={perception.intent}, Entities={perception.entities}")
        
        # Step 2: Decision
        decision = make_decision(
            perception=perception,
            url_memory=state.url_memory,
            search_history=state.search_history.get_recent_searches()
        )
        print(f"[Agent] Decision: Action={decision.action}, Reasoning={decision.reasoning}")
        
        # Step 3: Action
        result = await state.action_executor.execute(
            action_name=decision.action,
            parameters=decision.parameters
        )
        print(f"[Agent] Action result: Success={result.success}")
        
        # Step 4: Return structured response
        return {
            "query": request.query,
            "perception": perception.model_dump(),
            "decision": decision.model_dump(),
            "action_result": {
                "success": result.success,
                "action": result.action_name,
                "result": result.result,
                "message": result.message
            }
        }
    
    except Exception as e:
        print(f"[Agent] Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Main ====================

def log(stage: str, message: str):
    """Log messages"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [Agent] [{stage}] {message}")


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    log("startup", "RAG Chrome Plugin Agent starting...")
    log("startup", f"Index loaded: {state.index_manager.index.ntotal if state.index_manager.index else 0} vectors")
    log("startup", f"URL memory: {len(state.url_memory.url_memory)} URLs tracked")
    log("startup", "Agent ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    log("shutdown", "Saving state...")
    state.index_manager._save_index()
    log("shutdown", "Agent stopped.")


if __name__ == "__main__":
    log("main", "Starting FastAPI server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

