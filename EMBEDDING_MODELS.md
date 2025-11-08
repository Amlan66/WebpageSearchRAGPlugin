# Embedding Models - RAG Chrome Plugin

## 🎯 Model Choice: nomic-embed-text

### Why nomic-embed-text?

**nomic-embed-text** is used for both local (Ollama) and Colab (Sentence Transformers) to ensure **perfect consistency** in embeddings across the system.

---

## 📊 Model Specifications

### nomic-embed-text (v1.5)

| Property | Value |
|----------|-------|
| **Dimension** | 768 |
| **Context Length** | 8192 tokens |
| **Model Size** | ~500 MB |
| **Performance** | State-of-the-art on MTEB |
| **License** | Apache 2.0 (Open Source) |
| **Provider** | Nomic AI |

### Advantages:

1. ✅ **High Quality**: Top-tier performance on retrieval benchmarks
2. ✅ **Long Context**: 8192 tokens (vs 512 for many models)
3. ✅ **Consistent**: Same model locally and on Colab
4. ✅ **Open Source**: No API limits or costs
5. ✅ **Well Maintained**: Regular updates from Nomic AI

---

## 🔄 Where Each Model Runs

### Local System (Ollama)
```python
# EmbeddingService in toolsCP.py
embedding_service = EmbeddingService(
    local_url="http://localhost:11434/api/embeddings",
    model="nomic-embed-text"  # ← Uses Ollama
)

# Usage: Only for search query embeddings (single vector, fast)
query_embedding = embedding_service.get_embedding("user search query")
# Dimension: 768
```

**Why Ollama for queries?**
- ⚡ Fast for single embeddings (~100-200ms)
- 🏠 No network latency
- 🔒 Privacy (no data sent anywhere)

### Colab (Sentence Transformers)
```python
# colab_indexer.py
embedding_model = SentenceTransformer(
    'nomic-ai/nomic-embed-text-v1.5',
    trust_remote_code=True,
    device='cuda'  # ← Uses GPU
)

# Usage: Batch embedding for indexing (many vectors, needs speed)
embeddings = embedding_model.encode(
    chunks,  # List of 10-50 chunks
    batch_size=32,
    show_progress_bar=True
)
# Dimension: 768
```

**Why Sentence Transformers on Colab?**
- 🚀 GPU acceleration for batch processing
- 📦 Efficient batching (32 chunks at once)
- ⏱️ ~2-3 seconds for 50 chunks (vs 5-10 minutes locally)

---

## 🔍 Embedding Process

### Query Embedding (Local - Fast)
```
User search: "machine learning tutorials"
    ↓
Local Ollama (nomic-embed-text)
    ↓
Single vector (768-dim) in ~150ms
    ↓
Search FAISS index
```

### Document Embedding (Colab - GPU)
```
Webpage with 15 chunks
    ↓
Send to Colab
    ↓
Sentence Transformers (GPU)
    - Batch encode all 15 chunks
    - Model: nomic-embed-text
    - Output: 15 vectors × 768 dimensions
    ↓
Add to FAISS index
    ↓
~2-3 seconds total
```

---

## 📐 Dimension: 768

### What does 768 dimensions mean?

Each piece of text is converted to a vector of 768 floating-point numbers that capture semantic meaning.

**Example:**
```python
text = "Python is a programming language"
embedding = [0.023, -0.145, 0.891, ..., 0.234]  # 768 numbers
```

**Higher dimensions** = More nuanced semantic understanding
- 384-dim: Basic semantic similarity
- 768-dim: Rich semantic + context understanding ✅
- 1536-dim: Very detailed (but slower, more storage)

**nomic-embed-text (768)** strikes the perfect balance!

---

## 🔄 Consistency is Critical

### Why Same Model Everywhere?

```
❌ BAD: Different models
    Local: model-A (384-dim) 
    Colab: model-B (768-dim)
    Result: Incompatible! Search won't work!

✅ GOOD: Same model
    Local: nomic-embed-text (768-dim)
    Colab: nomic-embed-text (768-dim)
    Result: Perfect compatibility!
```

**Vector similarity only works if embeddings are in the same space!**

---

## 💾 Storage Impact

### Per Page Indexed:

**Text Data:**
- Chunks (text): ~5 KB
- Metadata: ~1 KB

**Embedding Data:**
- 15 chunks × 768 dimensions × 4 bytes = ~46 KB
- Total per page: ~52 KB

**100 Pages:**
- ~5.2 MB for embeddings
- ~600 KB for text/metadata
- **Total: ~5.8 MB**

**Very efficient storage!**

---

## 🚀 Performance Comparison

### Indexing Speed (50 chunks):

| Setup | Time | Hardware |
|-------|------|----------|
| Local (no GPU) | ~5-10 min | CPU only |
| Local (RTX 3060) | ~30 sec | Local GPU |
| **Colab (T4 GPU)** | **~2-3 sec** | Free GPU ✅ |
| Colab (A100 GPU) | ~1 sec | Paid GPU |

### Search Speed (query):

| Component | Time |
|-----------|------|
| Generate query embedding (local) | ~100-200ms |
| FAISS search (768-dim) | ~10-50ms |
| **Total** | **~150-250ms** |

**Fast enough for real-time search!**

---

## 🔧 Installation

### Local (Ollama)
```bash
# Install Ollama from https://ollama.ai

# Pull nomic-embed-text
ollama pull nomic-embed-text

# Test it
ollama run nomic-embed-text "Hello world"
```

### Colab (Sentence Transformers)
```python
# Already included in colab_indexer.py
!pip install -q sentence-transformers

# Model auto-downloads on first use
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')
```

---

## 🆚 Alternative Models (Not Used)

### Why not all-MiniLM-L6-v2?

| Feature | all-MiniLM-L6-v2 | nomic-embed-text |
|---------|------------------|------------------|
| Dimension | 384 | 768 ✅ |
| Context Length | 512 | 8192 ✅ |
| Quality | Good | Excellent ✅ |
| Speed | Very Fast | Fast ✅ |
| Size | 80 MB | 500 MB |

**Verdict**: nomic-embed-text is better quality for slightly lower speed (still fast!)

### Why not OpenAI embeddings?

| Feature | OpenAI | nomic-embed-text |
|---------|--------|------------------|
| Quality | Excellent | Excellent ✅ |
| Cost | $0.0001/1K tokens | Free ✅ |
| Privacy | Sends data to API | Local ✅ |
| Speed | Network latency | No network ✅ |
| Limit | Rate limited | No limits ✅ |

**Verdict**: nomic-embed-text is free, private, and fast!

---

## 📊 Compatibility Matrix

| Component | Model | Dimension | Compatible? |
|-----------|-------|-----------|-------------|
| Local Query Embedding | nomic-embed-text | 768 | ✅ |
| Colab Indexing | nomic-embed-text | 768 | ✅ |
| FAISS Index | IndexFlatL2 | 768 | ✅ |
| Search Results | - | 768 | ✅ |

**Perfect compatibility across the entire system!**

---

## 🔍 How to Verify Model

### Check Local Ollama:
```bash
ollama list
# Should show: nomic-embed-text

# Test embedding
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'

# Count dimensions in response
# Should be 768
```

### Check Colab:
```python
# In colab_indexer.py output
print(f"✅ Model loaded. Dimension: {EMBEDDING_DIM}")
# Should show: Dimension: 768

# Test embedding
test_embed = embedding_model.encode(["test"])
print(test_embed.shape)
# Should show: (1, 768)
```

---

## 🎯 Summary

- ✅ **Model**: nomic-embed-text (v1.5)
- ✅ **Dimension**: 768
- ✅ **Local**: Ollama (query embeddings)
- ✅ **Colab**: Sentence Transformers (batch indexing)
- ✅ **Consistency**: Same model everywhere
- ✅ **Performance**: Fast and high quality
- ✅ **Cost**: Free and open source

**Perfect choice for a production RAG system!** 🚀

