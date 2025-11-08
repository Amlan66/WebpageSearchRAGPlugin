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
└─────────────────────┘
```

## 📋 Prerequisites

### Local System
- Python 3.10+
- UV package manager
- Ollama with `nomic-embed-text` model (required for search query embeddings)
- Chrome browser

### Google Colab
- Google account with Colab access
- GPU runtime for faster indexing

## 🚀 Installation

### 1. Setup Local Agent

```bash
cd RAGChromePlugin

# Install all dependencies using UV (reads from pyproject.toml)
uv sync

# Install Ollama and pull embedding model for search query embeddings
# Download from https://ollama.ai
ollama pull nomic-embed-text
```

### 2. Configure Environment

Create a `.env` file in the `RAGChromePlugin` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 3. Setup Google Colab Indexer (Manual Workflow)

1. Open Google Colab: https://colab.research.google.com/
2. Upload `colab_indexer.py` to Colab
3. Change runtime to GPU:
   - Runtime → Change runtime type → GPU
4. Create `urls.txt` in Colab with URLs to index (one per line):
   ```python
   %%writefile urls.txt
   https://example.com/page1
   https://example.com/page2
   # Add your URLs here
   ```
5. Run the installation and indexing:
   ```python
   !pip install -q faiss-cpu sentence-transformers torch requests beautifulsoup4 lxml tqdm
   %run colab_indexer.py
   ```
6. Download the generated files:
   ```python
   from google.colab import files
   files.download('webpage_index/index.bin')
   files.download('webpage_index/metadata.json')
   files.download('webpage_index/url_index_cache.json')
   ```
7. Place downloaded files in `RAGChromePlugin/faiss_cache/` directory locally

### 4. Install Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `RAGChromePlugin/chrome_extension` folder
5. The extension should now appear in your toolbar

### 5. Start the Local Agent

```bash
cd RAGChromePlugin
python main.py
```

The server should start on `http://localhost:8000`

Now you're ready to search! Open the Chrome extension and try searching for content from your indexed pages.

## 📖 Usage

### Indexing Pages (Manual Colab Workflow)

1. **Collect URLs** you want to index
2. **Add them to `urls.txt`** in Google Colab (one URL per line)
3. **Run the indexer** in Colab:
   ```python
   %run colab_indexer.py
   ```
4. **Download the generated files**:
   - `index.bin` (FAISS vector index)
   - `metadata.json` (chunk metadata)
   - `url_index_cache.json` (content hashes for change detection)
5. **Place files in `RAGChromePlugin/faiss_cache/`** locally
6. **Restart the agent** if it's already running:
   ```bash
   python main.py
   ```

**What Gets Indexed:**
- ✅ Regular websites (news, blogs, documentation, etc.)
- ❌ Gmail, WhatsApp, YouTube (you can manually exclude these)
- ❌ Banking sites, login pages (recommended to exclude)

**Re-indexing Updated Content:**
- The indexer uses MD5 hashing to detect changed content
- Add the URL to `urls.txt` again and re-run the indexer
- Old chunks are automatically removed before adding new ones
- Download and replace the files in `faiss_cache/`

### Searching

1. **Click the extension icon**
2. **Type your query** in the search box
3. **Press Enter** - Automatically opens the top result and highlights text ✨
4. **OR click "Search"** - Shows top 3 results, click any to navigate

**Example Queries:**
- "authentication implementation details"
- "machine learning concepts explained"
- "pricing information"

### Features

- **Semantic Search**: Finds content by meaning, not just keywords
- **Auto-Navigation**: Press Enter to jump directly to the best match
- **Smart Highlighting**: Highlights relevant text in bright neon green
- **Top Results View**: Click search button to browse multiple results
- **Local Processing**: All searches happen locally for privacy and speed

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
- **LocalIndexManager**: Manages FAISS index (read-only, loads from faiss_cache/)
- **EmbeddingService**: Generates embeddings locally via Ollama (nomic-embed-text)
- **ColabClient**: Disabled (manual workflow only)
- Search tool for querying the local index

## 🔧 Configuration

### Chunking Strategy
Edit `colab_indexer.py`:
```python
CHUNK_SIZE = 256  # Words per chunk
CHUNK_OVERLAP = 40  # Overlapping words
```

### Excluded Domains
Manually exclude URLs in your `urls.txt` file. Do not add sensitive sites like:
- Gmail, WhatsApp, YouTube
- Banking and financial sites
- Login pages

### Embedding Model
Edit `colab_indexer.py` to change the model:
```python
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
```

## 📊 API Endpoints

### Local Agent (`localhost:8000`)

- `POST /search` - Search indexed content (main endpoint)
- `GET /status` - Get indexing statistics
- `POST /index` - Disabled (returns instructions for manual Colab indexing)
- Other endpoints available for extensibility

## 🐛 Troubleshooting

### Agent Not Connecting
```bash
# Check if agent is running
curl http://localhost:8000/

# Restart agent
python main.py
```

### Indexing Fails in Colab
- Ensure GPU runtime is selected
- Check if all dependencies are installed
- Verify `urls.txt` is uploaded and formatted correctly
- Check Colab logs for specific errors (403, network issues, etc.)

### Search Returns No Results
- Ensure you've indexed pages in Colab
- Verify downloaded files are in `faiss_cache/`:
  - `index.bin`
  - `metadata.json`
  - `url_index_cache.json`
- Restart the agent after placing files
- Check Ollama is running: `ollama list`

### Text Not Highlighting
- The indexed text may have changed on the webpage
- Try re-indexing the URL in Colab
- Check browser console (F12) for detailed logs
- The plugin tries multiple matching strategies automatically

### Extension Not Loading
- Check permissions in `chrome://extensions/`
- Verify you loaded the `chrome_extension` folder (not the root)
- Try reloading the extension (🔄 button)
- Check for errors in extension console

### Ollama Not Working
```bash
# Check if Ollama is installed and running
ollama list

# Pull the model if needed
ollama pull nomic-embed-text

# Test the model
ollama run nomic-embed-text "test query"
```

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

### Current Workflow (Manual Colab Indexing)

#### 1. Indexing Happens Only in Colab
- All webpage content fetching and embedding generation happens in Google Colab
- Uses GPU-accelerated `nomic-embed-text` model via Sentence Transformers
- Creates `index.bin`, `metadata.json`, and `url_index_cache.json`
- You manually download these files and place them in `faiss_cache/` locally

#### 2. Local Agent is Read-Only
- The local agent **only searches** the pre-built index
- It does NOT create or update the index
- Uses local Ollama (`nomic-embed-text`) only for query embeddings
- All indexing must be done through the Colab workflow

#### 3. Chrome Extension is the UI
**Hybrid approach**:
- Chrome extension: JavaScript (popup.js, content.js, background.js)
- Backend agent: Python (FastAPI with agentic components)
- Communication: HTTP REST API
- Extension sends search queries → Python searches FAISS → Extension displays results

#### 4. Change Detection with MD5 Hashing
- `url_index_cache.json` stores MD5 hash of each webpage's content
- When re-indexing, Colab checks if content changed
- If unchanged, skips re-indexing (saves time)
- If changed, removes old chunks and adds new ones (no duplicates)

## 🎯 Critical Design Decisions

1. **Manual Workflow**: No automatic syncing or ngrok - you control when indexing happens
2. **Colab-Only Indexing**: Heavy embedding work happens on GPU in Colab
3. **Local-Only Search**: Fast search with local FAISS and Ollama embeddings
4. **Change Detection**: MD5 hashing prevents duplicate entries and detects updates
5. **Chunking**: Same strategy as example3.py (256 words, 40 overlap)
6. **Smart Highlighting**: Multi-strategy text matching with neon green highlight
7. **Privacy First**: All data stays in your Colab and local machine

---

**Ready to search your browsing history intelligently!** 🚀

