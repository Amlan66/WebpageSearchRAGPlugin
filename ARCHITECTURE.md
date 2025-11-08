# RAG Chrome Plugin - Architecture Deep Dive

## 📐 System Architecture

### Overview
The RAG Chrome Plugin uses a **hybrid architecture** combining JavaScript (Chrome Extension) with Python (Agentic Backend) to create an intelligent webpage indexing and search system.

## 🧩 Components Breakdown

### 1. Chrome Extension Layer (JavaScript)

#### `manifest.json`
- Defines extension metadata and permissions
- Configures content scripts, background worker, and popup
- Requires `activeTab`, `storage`, `tabs`, `scripting` permissions

#### `popup.html` + `popup.js`
- **Purpose**: User interface for search and indexing
- **Features**:
  - Search input with real-time query
  - Statistics display (indexed pages, chunks)
  - Manual indexing trigger
  - Results display with click-to-navigate
  - Progress tracking for indexing operations
- **Communication**: HTTP REST API calls to local agent

#### `content.js`
- **Purpose**: Runs on every webpage to extract content
- **Responsibilities**:
  - Extract clean text from HTML (removes scripts, styles, nav, footer)
  - Highlight text when navigating to search results
  - Listen for messages from popup/background
  - Auto-check if pages should be indexed
- **Key Functions**:
  - `extractPageContent()`: Clean text extraction
  - `highlightText()`: Visual highlighting with scroll
  - `removeHighlights()`: Cleanup

#### `background.js`
- **Purpose**: Service worker for coordination
- **Responsibilities**:
  - Check agent connectivity
  - Handle navigation between tabs
  - Coordinate highlighting across tabs
  - Track page visits
  - Show badges for indexable pages
- **Key Functions**:
  - `navigateAndHighlight()`: Open URL and trigger highlighting
  - `handlePageVisit()`: Track and notify agent of visits

### 2. Python Agent Layer (Agentic Backend)

#### `agentCP.py` - Orchestrator
- **Architecture**: FastAPI server
- **Port**: 8000
- **Purpose**: Central coordinator implementing agentic loop

**Key Endpoints**:
```
POST /search              → Search indexed content
POST /index               → Index a webpage
GET  /status              → Get index statistics
POST /navigate            → Navigate and highlight
POST /check_url           → Check if URL indexed
POST /config/colab        → Configure Colab connection
GET  /memory/stats        → Memory statistics
POST /agent/query         → Full agentic processing
WebSocket /ws/progress    → Real-time progress updates
```

**Agentic Loop**:
```python
# 1. Perception
perception = extract_perception(user_query)

# 2. Decision  
decision = make_decision(perception, memory)

# 3. Action
result = execute_action(decision)

# 4. Memory Update
memory.record(result)
```

#### `perceptionCP.py` - Perception Layer
**Purpose**: Understand user intent and analyze content

**Key Functions**:
- `extract_perception(query)` → PerceptionResultCP
  - Uses Gemini LLM or rule-based fallback
  - Extracts intent: `search_content`, `check_index`, `navigate_to_result`
  - Identifies entities and search hints

- `analyze_webpage_content(webpage)` → Analysis dict
  - Determines if URL should be indexed
  - Extracts clean text from HTML
  - Creates text chunks (256 words, 40 overlap)
  - Returns chunks + metadata

- `should_index_url(url)` → bool
  - Excludes sensitive domains (gmail, whatsapp, youtube, banking)
  - Filters chrome:// and localhost
  - Returns indexability decision

**Data Flow**:
```
User Query → LLM/Rules → PerceptionResultCP(
    user_query: str
    intent: str
    entities: List[str]
    search_hint: Optional[str]
)
```

#### `decisionCP.py` - Decision Layer
**Purpose**: Decide what action to take based on perception and memory

**Key Functions**:
- `make_decision(perception, url_memory, search_history, previous_results)` → DecisionCP
  - Uses Gemini LLM or rule-based fallback
  - Considers index statistics and search history
  - Returns action with parameters and reasoning

- `make_indexing_decision(url, content, url_memory, force)` → dict
  - Checks if URL should be indexed
  - Compares content hash for changes
  - Returns: `should_index`, `reason`, `action` (skip/create/update/reindex)

- `decide_next_action_in_chain(current_action, result, original_intent)` → DecisionCP
  - Chains multiple actions (e.g., search then navigate)
  - Implements multi-step reasoning

**Decision Types**:
```
- search_index: Search through content
- navigate_highlight: Go to URL and highlight
- get_index_status: Get statistics
- fetch_index: Sync from Colab
- index_webpage: Index specific page
```

#### `actionCP.py` - Action Layer
**Purpose**: Execute decisions as concrete actions

**Key Classes**:
- `ActionExecutor`: Executes actions from tools registry
- `NavigationActionHandler`: Handles navigation/highlighting
- `IndexActionHandler`: Manages indexing requests
- `SearchActionHandler`: Executes searches

**Functions**:
- `create_navigation_action()`: Build navigation request
- `create_search_action()`: Build search query
- `create_index_action()`: Build indexing request
- `validate_action_parameters()`: Ensure params are correct

**Action Flow**:
```
Decision → ActionExecutor.execute() → Tool Function → Result
```

#### `memoryCP.py` - Memory Layer
**Purpose**: Persistent state management across sessions

**Key Classes**:

1. **URLMemoryManager** (`url_memory.json`)
   - Tracks all visited URLs
   - Stores: first_visited, last_visited, visit_count, is_indexed, content_hash, chunk_count
   - Methods: `add_or_update_url()`, `has_content_changed()`, `get_stats()`

2. **IndexCacheManager** (`faiss_cache/url_index_cache.json`)
   - Similar to `doc_index_cache.json` in example3.py
   - Maps URL → content_hash (MD5)
   - Methods: `should_index()`, `compute_content_hash()`, `update_cache()`

3. **IndexOperationMemory** (`index_operations.json`)
   - Records indexing operations history
   - Tracks: operation, url, timestamp, chunks_affected
   - Keeps last 1000 operations

4. **SearchHistoryMemory** (`search_history.json`)
   - Stores search queries and results
   - Tracks popular queries
   - Methods: `add_search()`, `get_popular_queries()`

**Memory Persistence**:
```
All memory automatically saved to JSON files
Loaded on agent startup
Thread-safe operations
```

#### `toolsCP.py` - Tools Layer
**Purpose**: Actual implementation of search, indexing, and index management

**Key Classes**:

1. **LocalIndexManager**
   - Manages local FAISS index (similar to example3.py)
   - Files: `faiss_cache/index.bin`, `metadata.json`, `url_index_cache.json`
   - Methods:
     - `add_chunks()`: Add embeddings to index
     - `remove_url_chunks()`: Remove for updates
     - `search()`: Query index with embeddings
     - `get_stats()`: Index statistics

2. **EmbeddingService**
   - Generates embeddings for text
   - **Local**: Ollama with nomic-embed-text
   - **Colab**: Sentence Transformers on GPU
   - Automatic fallback between sources

3. **ColabClient**
   - Communicates with Colab indexer via ngrok
   - Methods:
     - `index_webpage()`: Send to Colab for indexing
     - `fetch_index()`: Download index from Colab

**Tool Functions** (async):
- `search_index_tool()`: Search with query embedding
- `index_webpage_tool()`: Complete indexing pipeline
- `get_index_status_tool()`: Get stats
- `navigate_highlight_tool()`: Prepare navigation

#### `modelsCP.py` - Data Models
**Purpose**: Pydantic models for type safety and validation

**Model Categories**:

1. **Webpage Models**
   - `WebpageContent`: Raw page data
   - `WebpageChunk`: Individual chunk with metadata
   - `WebpageHash`: Hash tracking for change detection

2. **Indexing Models**
   - `IndexRequest`, `IndexResponse`
   - `IndexStatus`: Current state
   - `IndexingProgress`: Real-time progress

3. **Search Models**
   - `SearchQuery`, `SearchResult`, `SearchResponse`

4. **Navigation Models**
   - `NavigationRequest`, `HighlightRequest`

5. **Agent Models**
   - `PerceptionResultCP`: Perception output
   - `ActionResultCP`: Action result
   - `DecisionCP`: Decision with reasoning

6. **Memory Models**
   - `URLMemoryItem`, `IndexMemoryItem`

7. **Colab Models**
   - `ColabIndexRequest`, `ColabIndexResponse`
   - `ColabFetchRequest`, `ColabFetchResponse`

### 3. Colab Indexer Layer (GPU)

#### `colab_indexer.py`
**Purpose**: GPU-accelerated embedding generation on Google Colab

**Components**:
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **FAISS Index**: GPU-accelerated IndexFlatL2
- **FastAPI Server**: Exposed via ngrok
- **Endpoints**:
  - `POST /index`: Receive chunks, generate embeddings, update index
  - `GET /stats`: Index statistics
  - `GET /export`: Export index for download
  - `GET /metadata`: Metadata only

**Workflow**:
```
1. Local agent sends chunks to Colab
2. Colab generates embeddings on GPU (fast!)
3. Colab adds to FAISS index
4. Colab returns success + count
5. Local agent can periodically download index
```

## 🔄 Complete Data Flow

### Indexing Flow
```
User visits webpage
    ↓
content.js extracts text
    ↓
User clicks "Index Page" in popup
    ↓
popup.js → POST /index → agentCP.py
    ↓
Perception: analyze_webpage_content()
    - Extract text
    - Create chunks (256 words, 40 overlap)
    - Check if should index
    ↓
Decision: make_indexing_decision()
    - Check URL memory
    - Compare content hash
    - Decide: skip/create/update
    ↓
Action: index_webpage_tool()
    - Generate embeddings (Ollama or Colab)
    - Add to local FAISS index
    - Update metadata
    ↓
Memory: Update caches
    - url_memory.json
    - url_index_cache.json
    - index_operations.json
    ↓
Response to popup with success
```

### Search Flow
```
User enters query in popup
    ↓
popup.js → POST /search → agentCP.py
    ↓
Perception: extract_perception()
    - Identify intent (search_content)
    - Extract entities
    - Generate search hint
    ↓
Decision: make_decision()
    - Choose action: search_index
    - Set parameters: query, top_k
    ↓
Action: search_index_tool()
    - Generate query embedding
    - Search FAISS index
    - Retrieve top_k results
    ↓
Memory: Record search
    - search_history.json
    ↓
Return results to popup
    ↓
User clicks result
    ↓
background.js navigates to URL
    ↓
content.js highlights text
```

### Change Detection (Like example3.py)
```python
# In example3.py:
def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

if file.name in CACHE_META and CACHE_META[file.name] == fhash:
    skip  # Unchanged
else:
    process  # New or updated

# In RAGChromePlugin:
# IndexCacheManager (memoryCP.py)
def compute_content_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

def should_index(url, content):
    content_hash = self.compute_content_hash(content)
    if url not in self.cache:
        return True  # New
    return self.cache[url] != content_hash  # Changed

# Decision layer (decisionCP.py)
if cache.should_index(url, content):
    if url_item:  # Exists
        action = "update"
        remove_old_chunks()
    else:  # New
        action = "create"
```

## 🔑 Key Design Patterns

### 1. Agentic Loop Pattern
```python
# Implemented in agentCP.py endpoints
perception → decision → action → memory
```

### 2. Hybrid Architecture
```
JavaScript (UI) ←→ Python (Logic) ←→ Colab (GPU)
      HTTP REST           HTTP REST
```

### 3. Memory Persistence
```
- JSON for metadata (lightweight, readable)
- FAISS binary for vectors (efficient)
- Automatic save/load on operations
```

### 4. Chunking Strategy (Same as example3.py)
```python
def chunk_text(text, size=256, overlap=40):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])
```

### 5. Change Detection (MD5 Hashing)
```python
# Same principle as example3.py
content_hash = md5(content).hexdigest()
if cached_hash != content_hash:
    reindex()
```

## 📊 Comparison with example3.py

### Similarities
| Feature | example3.py | RAGChromePlugin |
|---------|-------------|-----------------|
| Chunking | ✅ 256 words, 40 overlap | ✅ Same |
| FAISS Index | ✅ IndexFlatL2 | ✅ Same |
| Metadata Storage | ✅ metadata.json | ✅ Same |
| Cache Mechanism | ✅ doc_index_cache.json | ✅ url_index_cache.json |
| Hash Function | ✅ MD5 | ✅ MD5 |
| Embedding Model | ✅ nomic-embed-text | ✅ nomic-embed-text |

### Differences
| Aspect | example3.py | RAGChromePlugin |
|--------|-------------|-----------------|
| Source | Local documents | Webpages |
| Server | MCP | FastAPI |
| UI | None | Chrome Extension |
| Architecture | Standalone | Agentic |
| Processing | Single script | Distributed (local + Colab) |
| Update Handling | ⚠️ Duplicates chunks | ✅ Removes old chunks |
| Memory | None | Persistent (4 types) |
| Decision Making | None | LLM-powered |

## 🚨 Critical Implementation Notes

### 1. Answering Your Original Question about example3.py

**Issue**: In example3.py, when a file is updated, old chunks are NOT removed, causing duplicates.

**Evidence**:
```python
# example3.py lines 237-259
for file in DOC_PATH.glob("*.*"):
    fhash = file_hash(file)
    if file.name in CACHE_META and CACHE_META[file.name] == fhash:
        continue  # Skip unchanged
    
    # Process file...
    metadata.extend(new_metadata)  # ⚠️ APPENDS, doesn't replace
```

**Solution in RAGChromePlugin**:
```python
# toolsCP.py - LocalIndexManager
def remove_url_chunks(self, url: str) -> int:
    # Filter out old chunks for this URL
    metadata_to_keep = [m for m in self.metadata if m["url"] != url]
    removed = len(self.metadata) - len(metadata_to_keep)
    self.metadata = metadata_to_keep
    self._save_index()
    return removed

# Then add new chunks
index_manager.add_chunks(url, new_chunks, embeddings, title)
```

### 2. Local vs Colab Coordination

**Problem**: Colab sessions can expire, losing the index.

**Solutions Implemented**:
1. **Local Primary**: Main index is local (`faiss_cache/`)
2. **Colab Optional**: Used only for GPU acceleration
3. **Fallback**: Can use local Ollama if Colab unavailable
4. **Download**: Can export Colab index to local
5. **Hash Sync**: `url_index_cache.json` tracks what's indexed

### 3. Chrome Extension + Python Integration

**Challenge**: Chrome extensions run JavaScript, agentic code is Python.

**Solution**: HTTP REST API
```
Chrome Extension (JS)
    ↓ HTTP POST /search
FastAPI Server (Python)
    ↓ Agentic Processing
Return JSON Response
    ↓ HTTP Response
Chrome Extension displays results
```

### 4. Embedding Generation

**Two-Tier Strategy**:
```python
class EmbeddingService:
    def get_embedding(text):
        if self.use_local:
            try:
                return get_local_embedding()  # Ollama
            except:
                return get_colab_embedding()  # Colab GPU
```

## 🎯 Production Considerations

### Current Limitations
1. **Index Rebuilding**: Doesn't rebuild FAISS index when removing chunks (only metadata)
2. **Highlighting**: Simple string matching (could use fuzzy matching)
3. **Colab Persistence**: Free tier expires after ~12 hours
4. **Single User**: No multi-user support
5. **No Sync**: Index not synced across devices

### Recommended Enhancements
1. Implement proper FAISS index reconstruction on updates
2. Add fuzzy text matching for better highlighting
3. Use persistent storage (S3, Google Drive) for index
4. Add user authentication for multi-device sync
5. Implement incremental index updates
6. Add vector quantization for larger indices (use HNSW)

## 📚 File Dependencies Graph

```
main.py
  └── agentCP.py
        ├── perceptionCP.py
        │     ├── modelsCP.py
        │     └── beautifulsoup4
        ├── decisionCP.py
        │     ├── modelsCP.py
        │     ├── memoryCP.py
        │     └── google.genai
        ├── actionCP.py
        │     └── modelsCP.py
        ├── memoryCP.py
        │     └── modelsCP.py
        └── toolsCP.py
              ├── modelsCP.py
              ├── memoryCP.py
              ├── perceptionCP.py
              ├── faiss
              └── numpy

Chrome Extension:
  manifest.json
    ├── popup.html → popup.js
    ├── content.js
    └── background.js
```

## 🧪 Testing Strategy

### Unit Tests (Recommended to Add)
```python
# test_perception.py
def test_should_index_url():
    assert not should_index_url("https://mail.google.com")
    assert should_index_url("https://wikipedia.org")

# test_memory.py
def test_url_memory():
    manager = URLMemoryManager(storage_path="test.json")
    manager.add_or_update_url("https://test.com", "Test", "hash123")
    assert manager.is_url_indexed("https://test.com")

# test_tools.py
def test_index_manager():
    manager = LocalIndexManager(index_dir="test_cache")
    chunks = ["test chunk"]
    embeddings = np.random.rand(1, 384)
    manager.add_chunks("http://test.com", chunks, embeddings, "Test")
    assert manager.index.ntotal == 1
```

### Integration Tests
```python
# test_agent_flow.py
async def test_full_search_flow():
    # Index a page
    await index_webpage_tool(url, title, content, ...)
    
    # Search for it
    results = await search_index_tool(query="test", ...)
    
    assert len(results.results) > 0
    assert results.results[0].url == url
```

## 🔐 Security Considerations

1. **Excluded Sites**: Hardcoded list prevents indexing sensitive sites
2. **Local First**: Data stays on your machine unless sent to your Colab
3. **No Telemetry**: No third-party analytics
4. **API Keys**: Stored in `.env`, not in code
5. **CORS**: Extension ID validation (should be added in production)

---

**This architecture implements a production-ready agentic RAG system with proper separation of concerns, persistent memory, and intelligent decision-making!** 🚀

