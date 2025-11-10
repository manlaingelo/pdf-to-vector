# TODO: Migration to Self-Hosted Embeddings

## Objective
Migrate from Google's text-embedding-004 API to local SentenceTransformer model for handling confidential data securely.

## Current State
- ✅ Using Google text-embedding-004 (cloud-based)
- ⚠️ **Security Risk**: Confidential data sent to Google servers
- Files affected: `cluster_pdfs.py`, `rag_chat.py`

## Migration Plan

### 1. Install Required Dependencies
```bash
pip install sentence-transformers torch
```

**Minimum Requirements for `all-MiniLM-L6-v2`:**
- **Disk Space**: ~80-100 MB (model size)
- **RAM**: 2-4 GB minimum
- **CPU**: Any modern CPU (multi-core recommended)
- **GPU** (optional): CUDA-compatible GPU for faster inference
- **Python**: 3.8+
- **Dependencies**: PyTorch, sentence-transformers

### 2. Update `cluster_pdfs.py`

Replace the `GoogleEmbeddingFunction` class:

```python
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        # Model will be downloaded on first run (~80MB)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input_data):
        # Returns embeddings as list of lists
        embeddings = self.model.encode(input_data, convert_to_numpy=True)
        return embeddings.tolist()
```

Update the collection creation:
```python
collection = client.get_or_create_collection(
    name=name,
    embedding_function=LocalEmbeddingFunction()  # Changed from GoogleEmbeddingFunction
)
```

**Remove/Update:**
- Remove `from google import genai`
- Remove `GEMINI_API_KEY` requirement
- Update `EMBEDDING_MODEL` constant to `"all-MiniLM-L6-v2"` (for reference)

### 3. Update `rag_chat.py`

Apply the same `LocalEmbeddingFunction` class:

```python
from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input_data):
        embeddings = self.model.encode(input_data, convert_to_numpy=True)
        return embeddings.tolist()
```

Update collection retrieval:
```python
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=LocalEmbeddingFunction()  # Changed
)
```

**Note**: Keep Gemini API for text generation (if acceptable), or replace with local LLM.

### 4. Alternative Models (Consider if needed)

If `all-MiniLM-L6-v2` doesn't meet requirements:

| Model | Size | Embedding Dim | Performance | Use Case |
|-------|------|---------------|-------------|----------|
| `all-MiniLM-L6-v2` | 80MB | 384 | Fast, good baseline | General purpose |
| `all-mpnet-base-v2` | 420MB | 768 | Better quality | Higher accuracy needed |
| `paraphrase-MiniLM-L6-v2` | 80MB | 384 | Semantic similarity | Paraphrase detection |
| `multi-qa-MiniLM-L6-cos-v1` | 80MB | 384 | QA optimized | RAG/Q&A systems |

### 5. Database Migration

⚠️ **Important**: Existing ChromaDB embeddings are incompatible with new model!

**Option A - Fresh Start (Recommended):**
```bash
rm -rf ./chroma_db
python cluster_pdfs.py  # Re-embed everything
```

**Option B - Backup & Migrate:**
```bash
cp -r ./chroma_db ./chroma_db.backup
rm -rf ./chroma_db
python cluster_pdfs.py  # Re-embed with new model
```

### 6. Testing Checklist

- [ ] Install sentence-transformers successfully
- [ ] Model downloads on first run (~80MB)
- [ ] `cluster_pdfs.py` runs without Google API key
- [ ] ChromaDB collection created with local embeddings
- [ ] K-Means clustering completes successfully
- [ ] `rag_chat.py` retrieves relevant documents
- [ ] Query results are semantically meaningful
- [ ] No data sent to external APIs (verify network logs if needed)

### 7. Performance Considerations

**Embedding Speed:**
- CPU: ~50-200 docs/second
- GPU (CUDA): ~500-2000 docs/second

**Optimization tips:**
- Use batch encoding: `model.encode(texts, batch_size=32)`
- Enable GPU if available: model automatically uses CUDA
- Cache embeddings in ChromaDB (already implemented)

### 8. Optional: Complete Local Setup

To run 100% offline (including LLM generation):

```bash
pip install llama-cpp-python
# Download a local model like Llama 2, Mistral, or Phi
```

Replace Gemini generation in `rag_chat.py` with local LLM.

## Timeline
- **Phase 1**: Install dependencies & test model (30 min)
- **Phase 2**: Update code (1-2 hours)
- **Phase 3**: Re-embed database (depends on PDF count)
- **Phase 4**: Testing & validation (1 hour)

## Risks & Mitigation
- **Risk**: Lower embedding quality vs Google's model
  - **Mitigation**: Test with sample queries; consider `all-mpnet-base-v2` if needed
- **Risk**: Slower embedding generation
  - **Mitigation**: Use GPU or accept slower processing for security
- **Risk**: Database re-embedding required
  - **Mitigation**: Backup existing database before migration

## Success Criteria
✅ No confidential data sent to external APIs  
✅ All functionality works as before  
✅ Acceptable query relevance and clustering quality  
✅ Documentation updated  

---

**Priority**: HIGH (Security & Compliance)  
**Effort**: MEDIUM (2-4 hours)  
**Status**: PENDING
