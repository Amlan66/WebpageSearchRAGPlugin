"""
Action module for RAG Chrome Plugin
Executes actions like navigation, highlighting, indexing requests
"""

from typing import Dict, Any, Optional
from modelsCP import (
    ActionResultCP, NavigationRequest, NavigationResponse,
    HighlightRequest, SearchQuery, SearchResponse,
    IndexRequest, IndexResponse, WebpageContent
)
import datetime


def log_action(stage: str, message: str):
    """Log action events"""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [Action] [{stage}] {message}")


class ActionExecutor:
    """Executes actions based on decisions"""
    
    def __init__(self, tools_registry: Dict[str, Any]):
        """
        Initialize action executor with tools
        
        Args:
            tools_registry: Dictionary of tool_name -> tool_function
        """
        self.tools = tools_registry
    
    async def execute(self, action_name: str, parameters: Dict[str, Any]) -> ActionResultCP:
        """
        Execute an action with given parameters
        
        Args:
            action_name: Name of the action/tool to execute
            parameters: Parameters to pass to the action
            
        Returns:
            ActionResultCP with execution result
        """
        log_action("execute", f"Action: {action_name}, Params: {parameters}")
        
        if action_name not in self.tools:
            return ActionResultCP(
                action_name=action_name,
                success=False,
                result=None,
                message=f"Action '{action_name}' not found in tools registry"
            )
        
        try:
            tool_func = self.tools[action_name]
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)
            
            log_action("success", f"{action_name} completed: {result}")
            
            return ActionResultCP(
                action_name=action_name,
                success=True,
                result=result,
                message=f"Successfully executed {action_name}"
            )
        
        except Exception as e:
            log_action("error", f"{action_name} failed: {str(e)}")
            return ActionResultCP(
                action_name=action_name,
                success=False,
                result=None,
                message=f"Error executing {action_name}: {str(e)}"
            )


# Import asyncio for async execution
import asyncio


def create_navigation_action(url: str, chunk_text: str, chunk_id: str) -> NavigationRequest:
    """
    Create a navigation request action
    
    Args:
        url: Target URL to navigate to
        chunk_text: Text to highlight on the page
        chunk_id: ID of the chunk for reference
        
    Returns:
        NavigationRequest object
    """
    return NavigationRequest(
        url=url,
        chunk_text=chunk_text,
        chunk_id=chunk_id
    )


def create_highlight_action(url: str, text_to_highlight: str, chunk_id: Optional[str] = None) -> HighlightRequest:
    """
    Create a highlight request action
    
    Args:
        url: URL where text should be highlighted
        text_to_highlight: The text to highlight
        chunk_id: Optional chunk ID for reference
        
    Returns:
        HighlightRequest object
    """
    return HighlightRequest(
        url=url,
        text_to_highlight=text_to_highlight,
        chunk_id=chunk_id
    )


def create_search_action(query: str, top_k: int = 3, url_filter: Optional[str] = None) -> SearchQuery:
    """
    Create a search query action
    
    Args:
        query: Search query text
        top_k: Number of results to return
        url_filter: Optional URL pattern to filter results
        
    Returns:
        SearchQuery object
    """
    return SearchQuery(
        query=query,
        top_k=top_k,
        url_filter=url_filter
    )


def create_index_action(
    url: str, 
    title: str, 
    content: str, 
    force_reindex: bool = False
) -> IndexRequest:
    """
    Create an indexing request action
    
    Args:
        url: URL to index
        title: Page title
        content: Page content (HTML or text)
        force_reindex: Force reindexing even if already indexed
        
    Returns:
        IndexRequest object
    """
    return IndexRequest(
        url=url,
        title=title,
        content=content,
        force_reindex=force_reindex
    )


class NavigationActionHandler:
    """Handles navigation and highlighting actions"""
    
    @staticmethod
    def navigate_and_highlight(request: NavigationRequest) -> NavigationResponse:
        """
        Navigate to URL and prepare highlight information
        This will be sent to Chrome extension to execute
        
        Args:
            request: NavigationRequest with URL and text to highlight
            
        Returns:
            NavigationResponse with status
        """
        log_action("navigate", f"URL: {request.url}, Chunk: {request.chunk_id}")
        
        try:
            # In actual implementation, this sends message to Chrome extension
            # For now, we prepare the response
            return NavigationResponse(
                success=True,
                url=request.url,
                message=f"Navigation prepared for {request.url}"
            )
        except Exception as e:
            return NavigationResponse(
                success=False,
                url=request.url,
                message=f"Navigation failed: {str(e)}"
            )
    
    @staticmethod
    def highlight_text(request: HighlightRequest) -> Dict[str, Any]:
        """
        Prepare text highlighting information
        
        Args:
            request: HighlightRequest with URL and text to highlight
            
        Returns:
            Dictionary with highlight information
        """
        log_action("highlight", f"URL: {request.url}, Text length: {len(request.text_to_highlight)}")
        
        return {
            "action": "highlight",
            "url": request.url,
            "text": request.text_to_highlight,
            "chunk_id": request.chunk_id,
            "timestamp": datetime.datetime.now().isoformat()
        }


class IndexActionHandler:
    """Handles indexing request actions"""
    
    @staticmethod
    async def request_indexing(
        request: IndexRequest,
        colab_client: Any,
        memory_manager: Any
    ) -> IndexResponse:
        """
        Request indexing of a webpage through Colab
        
        Args:
            request: IndexRequest with page information
            colab_client: Client to communicate with Colab
            memory_manager: Memory manager to track indexing
            
        Returns:
            IndexResponse with indexing status
        """
        log_action("index_request", f"URL: {request.url}, Force: {request.force_reindex}")
        
        try:
            # Check if already indexed and not forcing reindex
            if not request.force_reindex:
                if memory_manager.url_memory.is_url_indexed(request.url):
                    content_hash = memory_manager.index_cache.compute_content_hash(request.content)
                    if not memory_manager.index_cache.should_index(request.url, request.content):
                        return IndexResponse(
                            success=True,
                            url=request.url,
                            chunks_created=0,
                            message="URL already indexed with same content",
                            was_updated=False
                        )
            
            # Send to Colab for indexing (placeholder - will be implemented in agentCP)
            log_action("index", f"Sending to Colab: {request.url}")
            
            # This will be implemented with actual Colab communication
            return IndexResponse(
                success=True,
                url=request.url,
                chunks_created=0,  # Will be filled by actual indexing
                message="Indexing request prepared",
                was_updated=request.force_reindex
            )
        
        except Exception as e:
            log_action("error", f"Indexing failed for {request.url}: {str(e)}")
            return IndexResponse(
                success=False,
                url=request.url,
                chunks_created=0,
                message=f"Indexing failed: {str(e)}",
                was_updated=False
            )


class SearchActionHandler:
    """Handles search actions"""
    
    @staticmethod
    async def execute_search(
        query: SearchQuery,
        search_tool: Any
    ) -> SearchResponse:
        """
        Execute a search query
        
        Args:
            query: SearchQuery object
            search_tool: Search tool function
            
        Returns:
            SearchResponse with results
        """
        log_action("search", f"Query: '{query.query}', top_k: {query.top_k}")
        
        try:
            # Execute search using the tool
            results = await search_tool(
                query=query.query,
                top_k=query.top_k,
                url_filter=query.url_filter
            )
            
            log_action("search_complete", f"Found {len(results.results)} results")
            return results
        
        except Exception as e:
            log_action("error", f"Search failed: {str(e)}")
            return SearchResponse(
                query=query.query,
                results=[],
                total_found=0
            )


def format_action_result(result: ActionResultCP, verbose: bool = True) -> str:
    """
    Format action result for display
    
    Args:
        result: ActionResultCP object
        verbose: Include detailed information
        
    Returns:
        Formatted string
    """
    status = "✅" if result.success else "❌"
    output = f"{status} {result.action_name}: {result.message}"
    
    if verbose and result.result:
        output += f"\n  Result: {result.result}"
    
    return output


def validate_action_parameters(action_name: str, parameters: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that action parameters are correct
    
    Args:
        action_name: Name of the action
        parameters: Parameters to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_params = {
        "search_index": ["query"],
        "navigate_highlight": ["url", "chunk_text", "chunk_id"],
        "index_webpage": ["url", "title", "content"],
        "fetch_index": [],
        "get_index_status": []
    }
    
    if action_name not in required_params:
        return False, f"Unknown action: {action_name}"
    
    required = required_params[action_name]
    missing = [p for p in required if p not in parameters]
    
    if missing:
        return False, f"Missing required parameters: {missing}"
    
    return True, ""

