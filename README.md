# RAG Chrome Plugin

An intelligent Chrome extension that indexes and searches your browsing history using RAG (Retrieval-Augmented Generation) with an agentic architecture.

## 🌟 Features

- **Intelligent Webpage Indexing**: Automatically indexes webpages you visit (excluding sensitive sites)
- **Semantic Search**: Search across all indexed pages using natural language
- **Smart Highlighting**: Automatically highlights relevant text when navigating to search results
- **Agentic Architecture**: Uses perception, decision, action, and memory components
- **GPU-Accelerated**: Indexing happens on Google Colab with GPU for fast embedding generation
- **Privacy-Focused**: Excludes Gmail, WhatsApp, YouTube, and other sensitive sites

## 🏗️ Architecture

```
┌─────────────────────┐
│  Chrome Extension   │
│   (JavaScript)      │
└──────────┬──────────┘
           │
           ├── Captures page content
           ├── Sends search queries
           └── Receives results
           │
┌──────────▼──────────┐
│   Local Agent       │
│   (FastAPI/Python)  │
├─────────────────────┤
│ • perceptionCP.py   │ ← Analyzes user intent
│ • decisionCP.py     │ ← Decides actions
│ • actionCP.py       │ ← Executes actions
│ • memoryCP.py       │ ← Tracks state
│ • toolsCP.py        │ ← Local tools
└──────────┬──────────┘
           │
           ├── Manages local FAISS index
           ├── Handles search queries
           └── Coordinates with Colab
           │
┌──────────▼──────────┐
│  Google Colab       │
│  (GPU Indexing)     │
├─────────────────────┤
│ • Creates embeddings│
│ • Updates FAISS idx │
│ • Exposed via ngrok │
└─────────────────────┘
```

## 📋 Prerequisites

### Local System
- Python 3.10+
- UV package manager
- Ollama with `nomic-embed-text` model (optional, for local embeddings)
- Chrome browser

### Google Colab
- Google account with Colab access
- ngrok account (free tier works)

## 🚀 Installation

### 1. Setup Local Agent

```bash
cd RAGChromePlugin

# Install all dependencies using UV (reads from pyproject.toml)
uv sync

# Optional: Install Ollama and pull embedding model for local embeddings
# Download from https://ollama.ai
ollama pull nomic-embed-text
```

### 2. Configure Environment

Create a `.env` file in the `RAGChromePlugin` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 3. Setup Google Colab Indexer

1. Open Google Colab: https://colab.research.google.com/
2. Upload `colab_indexer.py` to Colab
3. Change runtime to GPU:
   - Runtime → Change runtime type → GPU
4. Get ngrok auth token:
   - Sign up at https://dashboard.ngrok.com/
   - Copy your auth token
5. Edit `colab_indexer.py` and add your ngrok token:
   ```python
   NGROK_TOKEN = "your_ngrok_token_here"
   ```
6. Run the setup cells in Colab:
   ```python
   !pip install -q fastapi uvicorn pyngrok faiss-gpu numpy pydantic nest-asyncio sentence-transformers
   !python colab_indexer.py
   ```
7. Copy the ngrok URL that appears (e.g., `https://xxxx.ngrok.io`)

### 4. Install Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `chrome_extension` folder
5. The extension should now appear in your toolbar

### 5. Connect Everything

1. Start the local agent:
   ```bash
   cd RAGChromePlugin
   python agentCP.py
   ```
   
   The server should start on `http://localhost:8000`

2. Configure Colab connection:
   ```bash
   curl -X POST http://localhost:8000/config/colab \
        -H 'Content-Type: application/json' \
        -d '{"ngrok_url": "YOUR_NGROK_URL_FROM_COLAB"}'
   ```

3. Open the Chrome extension popup - you should see stats!

## 📖 Usage

### Indexing Pages

**Manual Indexing:**
1. Visit any webpage
2. Click the extension icon
3. Click "📄 Index Current Page"
4. Wait for indexing to complete (progress bar shows status)

**What Gets Indexed:**
- ✅ Regular websites (news, blogs, documentation, etc.)
- ❌ Gmail, WhatsApp, YouTube
- ❌ Banking sites
- ❌ Login pages

### Searching

1. Click the extension icon
2. Type your query in the search box
3. Press Enter or click "Search"
4. Click any result to navigate and highlight

**Example Queries:**
- "authentication implementation details"
- "machine learning concepts explained"
- "pricing information"

### Features

- **Smart Stats**: View indexed pages and chunks count
- **Current Page Status**: See if current page is indexed
- **One-Click Navigation**: Click results to jump to page with highlighted text

## 🧠 Agentic Components

### Perception (`perceptionCP.py`)
- Extracts user intent from search queries
- Analyzes webpage content
- Determines if URLs should be indexed
- Chunks text using overlapping windows

### Decision (`decisionCP.py`)
- Decides whether to search, index, or navigate
- Determines if content has changed (needs reindexing)
- Routes queries to appropriate tools
- Uses LLM (Gemini) for intelligent decisions

### Action (`actionCP.py`)
- Executes search operations
- Triggers indexing requests
- Handles navigation and highlighting
- Coordinates with Chrome extension

### Memory (`memoryCP.py`)
- **URLMemoryManager**: Tracks visited URLs and indexing state
- **IndexCacheManager**: Manages content hashes for change detection
- **IndexOperationMemory**: Records indexing history
- **SearchHistoryMemory**: Stores search queries

### Tools (`toolsCP.py`)
- **LocalIndexManager**: Manages FAISS index locally
- **EmbeddingService**: Generates embeddings (local or Colab)
- **ColabClient**: Communicates with Colab indexer
- Search, index, status, and navigation tools

## 🔧 Configuration

### Chunking Strategy
Edit `perceptionCP.py`:
```python
CHUNK_SIZE = 256  # Words per chunk
CHUNK_OVERLAP = 40  # Overlapping words
```

### Excluded Domains
Edit `perceptionCP.py` → `should_index_url()`:
```python
excluded_domains = [
    'mail.google.com',
    'gmail.com',
    # Add more...
]
```

### Index Sync Interval
Edit `decisionCP.py`:
```python
sync_interval_minutes = 30  # How often to sync with Colab
```

## 📊 API Endpoints

### Local Agent (`localhost:8000`)

- `POST /search` - Search indexed content
- `POST /index` - Index a webpage
- `GET /status` - Get indexing statistics
- `POST /navigate` - Navigate and highlight
- `POST /check_url` - Check if URL is indexed
- `POST /config/colab` - Configure Colab URL
- `GET /memory/stats` - Get memory statistics
- `POST /agent/query` - Full agentic query processing

### Colab Indexer

- `GET /` - Health check
- `POST /index` - Index webpage (with chunks)
- `GET /stats` - Get index statistics
- `GET /export` - Export complete index
- `GET /metadata` - Get metadata only

## 🐛 Troubleshooting

### Agent Not Connecting
```bash
# Check if agent is running
curl http://localhost:8000/

# Restart agent
python agentCP.py
```

### Colab Session Expired
1. Go back to Colab notebook
2. Run the server cell again
3. Copy new ngrok URL
4. Reconfigure: `POST /config/colab`

### Indexing Fails
- Check Ollama is running: `ollama list`
- Verify Gemini API key in `.env`
- Check Colab ngrok URL is configured
- View agent logs in terminal

### Search Returns No Results
- Index some pages first
- Check index stats in extension popup
- Verify FAISS index exists: `ls faiss_cache/`

### Extension Not Loading Pages
- Check permissions in `chrome://extensions/`
- Verify content script is injected (check console)
- Try reloading the extension

## 📁 Project Structure

```
RAGChromePlugin/
├── agentCP.py              # FastAPI server (orchestrator)
├── perceptionCP.py         # Intent analysis & content extraction
├── decisionCP.py           # Decision making logic
├── actionCP.py             # Action execution
├── memoryCP.py             # Memory management
├── toolsCP.py              # Local tools & FAISS index
├── modelsCP.py             # Pydantic models
├── colab_indexer.py        # Colab GPU indexing script
├── pyproject.toml          # UV dependencies
├── .env                    # Environment variables
├── faiss_cache/            # Local FAISS index storage
│   ├── index.bin
│   ├── metadata.json
│   └── url_index_cache.json
└── chrome_extension/       # Chrome extension files
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── content.js
    ├── background.js
    └── icons/
```

## 🔒 Privacy & Security

- **No data leaves your control** except to Colab (which you manage)
- Sensitive sites automatically excluded from indexing
- All data stored locally or in your Colab session
- No third-party analytics or tracking
- Source code is fully auditable

## 🤝 Contributing

This is a reference implementation for educational purposes. Feel free to:
- Modify chunking strategies
- Add new excluded domains
- Improve highlighting algorithm
- Enhance UI/UX
- Optimize indexing performance

## 📝 Known Limitations

1. **Colab Session Timeout**: Free Colab sessions expire after ~12 hours of inactivity
   - **Solution**: Download index periodically or use paid Colab
   
2. **Highlighting Accuracy**: Text highlighting uses simple string matching
   - **Solution**: Implement fuzzy matching or better text search

3. **Chrome Only**: Currently only works with Chrome/Chromium browsers
   - **Solution**: Port to Firefox with WebExtensions

4. **Index Updates**: Updating existing pages doesn't remove old vectors efficiently
   - **Solution**: Implement proper index reconstruction

5. **No Cross-Device Sync**: Index is local to your machine
   - **Solution**: Add cloud storage for index (S3, Google Drive, etc.)

## 🔮 Future Enhancements

- [ ] Better duplicate detection in FAISS index
- [ ] Support for images and PDFs
- [ ] Multi-language support
- [ ] Export/import index functionality
- [ ] Better highlighting with fuzzy matching
- [ ] Auto-sync index across devices
- [ ] Query suggestions based on history
- [ ] Summarization of search results
- [ ] Collections/tags for organizing pages

## 📄 License

MIT License - Feel free to use and modify as needed.

## 🙏 Acknowledgments

- Built on FastAPI, FAISS, and Sentence Transformers
- Inspired by agentic architecture patterns
- Uses Gemini for decision-making intelligence
- Chrome Extension APIs for browser integration

## 💡 Key Insights

### Answering Your Doubts

#### 1. Do we need MCP server running locally?
**No**, not for this implementation. We replaced MCP with FastAPI for simpler HTTP communication between Chrome extension and Python backend. The agentic components are integrated directly into FastAPI.

#### 2. Handling Colab GPU accessibility
**Solution implemented**:
- Index is managed locally in `faiss_cache/`
- Colab is optional for GPU-accelerated indexing
- Falls back to local Ollama embeddings if Colab unavailable
- Can download index from Colab to local periodically
- `url_index_cache.json` hash checking determines what needs updating

#### 3. Chrome plugins with Python?
**Hybrid approach**:
- Chrome extension: JavaScript (popup.js, content.js, background.js)
- Backend agent: Python (agentic components)
- Communication: HTTP REST API
- Extension sends page content → Python processes → Extension displays results

#### 4. Integrating nomic-embed-text
**Two options**:
- **Local**: Ollama with nomic-embed-text (in EmbeddingService)
- **Colab**: Sentence Transformers on GPU (faster)
- Agent automatically tries local first, can fall back to Colab

## 🎯 Critical Design Decisions

1. **Local vs Colab Indexing**: Local agent manages search, Colab handles heavy embedding generation
2. **Change Detection**: MD5 hashing of content (same as example3.py)
3. **Chunking**: Same strategy as example3.py (256 words, 40 overlap)
4. **Memory Persistence**: JSON files for state, FAISS binary for vectors
5. **Excluded Sites**: Hardcoded list in perception layer

---

**Ready to search your browsing history intelligently!** 🚀

