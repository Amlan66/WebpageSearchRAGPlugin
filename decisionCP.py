"""
Decision module for RAG Chrome Plugin
Makes decisions about what actions to take based on perception and memory
"""

import os
import re
from typing import List, Optional, Dict, Any
from modelsCP import PerceptionResultCP, DecisionCP, SearchResult
from memoryCP import URLMemoryManager, SearchHistoryMemory
from dotenv import load_dotenv
import datetime

# Try to use Gemini for advanced decision making, fall back to rule-based
try:
    from google import genai
    load_dotenv()
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
    if GEMINI_AVAILABLE:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except:
    GEMINI_AVAILABLE = False


def log_decision(stage: str, message: str):
    """Log decision events"""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [Decision] [{stage}] {message}")


def make_decision_rule_based(
    perception: PerceptionResultCP,
    url_memory: URLMemoryManager,
    search_history: Optional[List[Dict]] = None
) -> DecisionCP:
    """
    Rule-based decision making (fallback when Gemini unavailable)
    
    Args:
        perception: User's perceived intent and entities
        url_memory: Memory of visited URLs
        search_history: Recent search history
        
    Returns:
        DecisionCP with action and parameters
    """
    intent = perception.intent
    
    # Handle check_index intent
    if intent == 'check_index':
        log_decision("rule", "Intent: check_index -> get_index_status")
        return DecisionCP(
            action="get_index_status",
            parameters={},
            reasoning="User wants to check indexing status"
        )
    
    # Handle navigate_to_result intent
    if intent == 'navigate_to_result':
        log_decision("rule", "Intent: navigate_to_result -> need search first")
        # Need to search first to find what to navigate to
        return DecisionCP(
            action="search_index",
            parameters={
                "query": perception.search_hint or perception.user_query,
                "top_k": 1
            },
            reasoning="User wants to navigate, first searching for relevant content"
        )
    
    # Default: search_content intent
    log_decision("rule", "Intent: search_content -> search_index")
    return DecisionCP(
        action="search_index",
        parameters={
            "query": perception.search_hint or perception.user_query,
            "top_k": 3
        },
        reasoning="User wants to search through indexed content"
    )


def make_decision_llm(
    perception: PerceptionResultCP,
    url_memory: URLMemoryManager,
    search_history: Optional[List[Dict]] = None,
    previous_results: Optional[List[SearchResult]] = None
) -> DecisionCP:
    """
    LLM-based decision making using Gemini
    
    Args:
        perception: User's perceived intent and entities
        url_memory: Memory of visited URLs
        search_history: Recent search history
        previous_results: Previous search results if any
        
    Returns:
        DecisionCP with action and parameters
    """
    # Get context from memory
    stats = url_memory.get_stats()
    recent_searches = search_history[-5:] if search_history else []
    
    # Format previous results if available
    prev_results_text = ""
    if previous_results:
        prev_results_text = "\nPrevious search results:\n"
        for i, result in enumerate(previous_results[:3]):
            prev_results_text += f"  {i+1}. {result.title} ({result.url})\n"
    
    prompt = f"""
You are a decision-making AI for a RAG Chrome Plugin that indexes webpages.

User Query: "{perception.user_query}"
Intent: {perception.intent}
Entities: {', '.join(perception.entities)}
Search Hint: {perception.search_hint or 'None'}

Index Statistics:
- Total URLs indexed: {stats['indexed_urls']}
- Total chunks: {stats['total_chunks']}
- Unindexed URLs visited: {stats['unindexed_urls']}

Recent searches: {', '.join([s.get('query', '') for s in recent_searches]) if recent_searches else 'None'}
{prev_results_text}

Available actions:
1. search_index - Search through indexed webpage content
   Parameters: query (str), top_k (int, default 3), url_filter (optional str)
   
2. navigate_highlight - Navigate to a URL and highlight text
   Parameters: url (str), chunk_text (str), chunk_id (str)
   
3. get_index_status - Get current indexing statistics
   Parameters: none
   
4. fetch_index - Fetch latest index from Colab
   Parameters: none
   
5. index_webpage - Request indexing of a specific webpage
   Parameters: url (str), title (str), content (str)

Decide the BEST action to take and return a Python dictionary with:
- action: (one of the above action names)
- parameters: (dict with required parameters)
- reasoning: (brief explanation)

Rules:
- If intent is 'check_index', use get_index_status
- If intent is 'search_content', use search_index
- If intent is 'navigate_to_result' AND previous results exist, use navigate_highlight with the top result
- If intent is 'navigate_to_result' BUT no previous results, use search_index first
- If user asks about statistics or status, use get_index_status

Output ONLY the dictionary, no markdown or extra formatting.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        
        # Clean up response
        clean = re.sub(r"^```(?:json|python)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        
        # Parse response
        parsed = eval(clean)
        
        log_decision("llm", f"Action: {parsed['action']}, Reasoning: {parsed['reasoning']}")
        
        return DecisionCP(
            action=parsed['action'],
            parameters=parsed.get('parameters', {}),
            reasoning=parsed.get('reasoning', 'LLM decision')
        )
    
    except Exception as e:
        log_decision("error", f"LLM decision failed: {e}, falling back to rules")
        return make_decision_rule_based(perception, url_memory, search_history)


def make_decision(
    perception: PerceptionResultCP,
    url_memory: URLMemoryManager,
    search_history: Optional[List[Dict]] = None,
    previous_results: Optional[List[SearchResult]] = None
) -> DecisionCP:
    """
    Main decision-making function
    Uses LLM if available, otherwise uses rule-based approach
    
    Args:
        perception: User's perceived intent
        url_memory: Memory of visited URLs
        search_history: Recent search history
        previous_results: Previous search results if any
        
    Returns:
        DecisionCP with action to take
    """
    if GEMINI_AVAILABLE:
        return make_decision_llm(perception, url_memory, search_history, previous_results)
    else:
        return make_decision_rule_based(perception, url_memory, search_history)


def make_indexing_decision(
    url: str,
    content: str,
    url_memory: URLMemoryManager,
    force: bool = False
) -> Dict[str, Any]:
    """
    Decide whether to index a webpage and how
    
    Args:
        url: URL to potentially index
        content: Page content
        url_memory: Memory manager
        force: Force indexing regardless of cache
        
    Returns:
        Dictionary with decision details
    """
    from perceptionCP import should_index_url
    
    # Check if URL should be indexed at all
    if not should_index_url(url):
        log_decision("index", f"Skipping {url} - excluded domain")
        return {
            "should_index": False,
            "reason": "URL in excluded domains",
            "action": "skip"
        }
    
    # Check if URL is already indexed
    url_item = url_memory.get_url(url)
    
    if force:
        log_decision("index", f"Force reindex: {url}")
        return {
            "should_index": True,
            "reason": "Force reindex requested",
            "action": "reindex",
            "existing_chunks": url_item.chunk_count if url_item else 0
        }
    
    # Check if content has changed
    from memoryCP import IndexCacheManager
    cache = IndexCacheManager()
    
    if cache.should_index(url, content):
        if url_item:
            log_decision("index", f"Content changed: {url}")
            return {
                "should_index": True,
                "reason": "Content has changed since last index",
                "action": "update",
                "existing_chunks": url_item.chunk_count
            }
        else:
            log_decision("index", f"New URL: {url}")
            return {
                "should_index": True,
                "reason": "New URL not yet indexed",
                "action": "create"
            }
    else:
        log_decision("index", f"Already indexed and unchanged: {url}")
        return {
            "should_index": False,
            "reason": "URL already indexed with same content",
            "action": "skip"
        }


def decide_next_action_in_chain(
    current_action: str,
    current_result: Any,
    original_intent: str
) -> Optional[DecisionCP]:
    """
    Decide next action in a chain of actions
    
    Args:
        current_action: Action that was just executed
        current_result: Result from the action
        original_intent: Original user intent
        
    Returns:
        Next DecisionCP or None if chain is complete
    """
    # If we searched and user wanted to navigate, now we can navigate
    if current_action == "search_index" and original_intent == "navigate_to_result":
        if hasattr(current_result, 'results') and len(current_result.results) > 0:
            top_result = current_result.results[0]
            log_decision("chain", f"Search complete, now navigating to top result: {top_result.url}")
            return DecisionCP(
                action="navigate_highlight",
                parameters={
                    "url": top_result.url,
                    "chunk_text": top_result.chunk_text,
                    "chunk_id": top_result.chunk_id
                },
                reasoning="Navigating to top search result as user intended"
            )
    
    # If we fetched index and now have results, user likely wants to search
    if current_action == "fetch_index":
        log_decision("chain", "Index fetched, awaiting user search query")
        return None  # Wait for user's next query
    
    # Chain complete
    return None


def should_fetch_index_from_colab(
    last_sync_time: Optional[datetime.datetime],
    sync_interval_minutes: int = 30
) -> bool:
    """
    Decide if we should fetch updated index from Colab
    
    Args:
        last_sync_time: When index was last synced
        sync_interval_minutes: How often to sync
        
    Returns:
        True if should fetch, False otherwise
    """
    if last_sync_time is None:
        return True
    
    now = datetime.datetime.now()
    delta = now - last_sync_time
    
    should_sync = delta.total_seconds() / 60 > sync_interval_minutes
    
    if should_sync:
        log_decision("sync", f"Index is {delta.total_seconds()/60:.1f} minutes old, fetching update")
    
    return should_sync

