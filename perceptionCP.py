"""
Perception module for RAG Chrome Plugin
Analyzes user queries and webpage content to understand intent
"""

import re
from typing import Optional, List
from modelsCP import PerceptionResultCP, WebpageContent
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

# Try to use Gemini for advanced perception, fall back to rule-based
try:
    from google import genai
    load_dotenv()
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
    if GEMINI_AVAILABLE:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except:
    GEMINI_AVAILABLE = False


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"[Perception] HTML parsing error: {e}")
        return html_content


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 40) -> List[str]:
    """
    Chunk text into overlapping segments (same as example3.py)
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def should_index_url(url: str) -> bool:
    """
    Determine if a URL should be indexed based on patterns
    Excludes sensitive sites like gmail, whatsapp, youtube
    """
    # Sensitive domains to exclude
    excluded_domains = [
        'mail.google.com',
        'gmail.com',
        'web.whatsapp.com',
        'whatsapp.com',
        'youtube.com',
        'www.youtube.com',
        'accounts.google.com',
        'login.',
        'signin.',
        'auth.',
        'banking',
        'bank.',
        'paypal.com',
        'facebook.com/login',
        'twitter.com/login'
    ]
    
    url_lower = url.lower()
    
    # Check if URL contains any excluded domain
    for domain in excluded_domains:
        if domain in url_lower:
            return False
    
    # Exclude localhost and internal IPs
    if 'localhost' in url_lower or '127.0.0.1' in url_lower:
        return False
    
    # Exclude chrome:// and other browser internal pages
    if url_lower.startswith(('chrome://', 'about:', 'edge://', 'chrome-extension://')):
        return False
    
    return True


def extract_perception_rule_based(user_query: str) -> PerceptionResultCP:
    """
    Rule-based perception extraction (fallback when Gemini unavailable)
    """
    user_query_lower = user_query.lower()
    
    # Determine intent based on keywords
    if any(word in user_query_lower for word in ['search', 'find', 'look for', 'show me']):
        intent = 'search_content'
    elif any(word in user_query_lower for word in ['index', 'status', 'stats', 'how many']):
        intent = 'check_index'
    elif any(word in user_query_lower for word in ['go to', 'navigate', 'open', 'take me']):
        intent = 'navigate_to_result'
    else:
        intent = 'search_content'  # Default to search
    
    # Extract potential search terms (simple word extraction)
    entities = [
        word for word in user_query.split() 
        if len(word) > 3 and word.lower() not in ['what', 'where', 'when', 'search', 'find']
    ]
    
    return PerceptionResultCP(
        user_query=user_query,
        intent=intent,
        entities=entities,
        search_hint=user_query if intent == 'search_content' else None
    )


def extract_perception_llm(user_query: str) -> PerceptionResultCP:
    """
    LLM-based perception extraction using Gemini
    """
    prompt = f"""
You are analyzing a search query for a RAG Chrome Plugin that indexes webpages.

Query: "{user_query}"

Determine the user's intent from these options:
- 'search_content': User wants to search through indexed webpage content
- 'check_index': User wants to know indexing status or statistics
- 'navigate_to_result': User wants to navigate to a specific webpage

Return a Python dictionary with:
- intent: (one of the above)
- entities: list of important keywords/phrases from the query
- search_hint: (optimized search query if intent is search_content, else None)

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
        
        return PerceptionResultCP(
            user_query=user_query,
            intent=parsed.get('intent', 'search_content'),
            entities=parsed.get('entities', []),
            search_hint=parsed.get('search_hint')
        )
    
    except Exception as e:
        print(f"[Perception] LLM extraction failed: {e}, falling back to rules")
        return extract_perception_rule_based(user_query)


def extract_perception(user_query: str) -> PerceptionResultCP:
    """
    Main perception extraction function
    Uses LLM if available, otherwise uses rule-based approach
    """
    if GEMINI_AVAILABLE:
        return extract_perception_llm(user_query)
    else:
        return extract_perception_rule_based(user_query)


def analyze_webpage_content(webpage: WebpageContent) -> dict:
    """
    Analyze webpage content to extract metadata and determine indexability
    
    Returns:
        Dictionary with analysis results including:
        - should_index: bool
        - text_content: str (cleaned text)
        - chunks: List[str]
        - metadata: dict
    """
    # Check if URL should be indexed
    if not should_index_url(webpage.url):
        return {
            "should_index": False,
            "reason": "URL is in excluded domains list",
            "text_content": "",
            "chunks": [],
            "metadata": {}
        }
    
    # Extract text from HTML
    text_content = extract_text_from_html(webpage.content)
    
    # Check if content is substantial enough
    word_count = len(text_content.split())
    if word_count < 50:
        return {
            "should_index": False,
            "reason": "Content too short (< 50 words)",
            "text_content": text_content,
            "chunks": [],
            "metadata": {"word_count": word_count}
        }
    
    # Create chunks
    chunks = chunk_text(text_content)
    
    return {
        "should_index": True,
        "text_content": text_content,
        "chunks": chunks,
        "metadata": {
            "word_count": word_count,
            "chunk_count": len(chunks),
            "title": webpage.title,
            "url": webpage.url
        }
    }


def extract_search_context(query: str, nearby_text: str = "") -> str:
    """
    Enhance search query with context if available
    """
    if nearby_text:
        return f"{query} {nearby_text}"
    return query


# Utility function for logging
def log_perception(stage: str, message: str):
    """Log perception events"""
    import datetime
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [Perception] [{stage}] {message}")

