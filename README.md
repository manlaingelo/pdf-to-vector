# PDF to Vector - Document Clustering with ChromaDB

A Python application that extracts text from PDF files, generates embeddings using Google's Gemini API, stores them in ChromaDB, and performs K-Means clustering to organize documents by similarity.

## Features

- 📄 **PDF Text Extraction**: Extract text from all PDFs in a directory using PyPDF
- 🧠 **Smart Embeddings**: Generate vector embeddings using Google's `text-embedding-004` model
- 💾 **Vector Database**: Store and manage embeddings with ChromaDB
- 🎯 **Document Clustering**: Automatically cluster similar documents using K-Means
- 🔍 **Interactive Browser**: Browse and explore your document collection with built-in UI

## Prerequisites

- Python 3.11 or higher (< 3.14)
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key

## Installation

1. **Clone the repository**
   ```bash
   cd pdf-to-vector
   ```

2. **Install dependencies**
   ```bash
   make sync
   # or: uv sync
   ```

3. **Set up your Google API key**
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```
   
   Get your API key from [Google AI Studio](https://ai.google.dev/)

## Usage

### Quick Start with Makefile

```bash
# View all available commands
make help

# Process PDFs and create clusters
make cluster

# Browse the database
make browse

# Chat with your PDFs using AI
make chat

# Run the main script
make run
```

### Manual Usage

1. **Add your PDF files**
   
   Place PDF files in the `./pdfs` directory

2. **Process PDFs and create clusters**
   ```bash
   uv run python cluster_pdfs.py
   ```
   
   This will:
   - Extract text from all PDFs
   - Generate embeddings for each page
   - Store in ChromaDB at `./chroma_db`
   - Perform K-Means clustering (default: 10 clusters)
   - Update metadata with cluster assignments

3. **Browse the collection**
   ```bash
   uv run chroma browse pdf_cluster_data --local --path ./chroma_db
   # or: make browse
   ```

4. **Chat with your PDFs using RAG**
   ```bash
   # Interactive mode
   uv run python rag_chat.py
   # or: make chat
   
   # Single query mode
   uv run python rag_chat.py "What is this document about?"
   ```

## Configuration

Edit the constants in `cluster_pdfs.py`:

```python
PDF_DIR = "./pdfs"              # PDF input directory
CHROMA_PATH = "./chroma_db"     # ChromaDB storage path
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "text-embedding-004"
K_CLUSTERS = 10                 # Number of clusters
```

Edit the constants in `rag_chat.py`:

```python
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "text-embedding-004"
GENERATIVE_MODEL = "gemini-2.0-flash-exp"
TOP_K_RESULTS = 5               # Number of relevant documents to retrieve
RELEVANCE_THRESHOLD = 0.3       # Minimum relevance score (0-1)
```

## Project Structure

```
pdf-to-vector/
├── pdfs/                  # Place your PDF files here
├── chroma_db/            # ChromaDB persistent storage
├── cluster_pdfs.py       # Main clustering script
├── rag_chat.py           # RAG-powered chat interface
├── main.py               # Entry point script
├── browse.sh             # Helper script to browse DB
├── Makefile              # Task automation
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

## How It Works

### Document Processing (cluster_pdfs.py)

1. **Text Extraction**: PyPDF extracts text from each page of every PDF in the `pdfs/` directory

2. **Embedding Generation**: Each page's text is sent to Google's Gemini API to generate a vector embedding using the `text-embedding-004` model

3. **Storage**: Documents, embeddings, and metadata are stored in ChromaDB with unique IDs formatted as `{filename}_p{page_number}`

4. **Clustering**: K-Means algorithm groups similar documents based on their vector embeddings

5. **Metadata Update**: Each document is tagged with its `cluster_id` in ChromaDB

### RAG Chat System (rag_chat.py)

1. **Query Embedding**: User's question is converted to a vector using the same embedding model

2. **Semantic Search**: ChromaDB finds the most relevant document chunks based on vector similarity

3. **Relevance Filtering**: Documents below the relevance threshold are filtered out to prevent hallucinations

4. **Context Building**: Top-K relevant documents are formatted with metadata (source, page, cluster)

5. **AI Generation**: Gemini generates a contextual answer using the retrieved documents

6. **Response**: AI provides an answer with source citations, or a "no data" message if the question is out of scope

## Dependencies

- **chromadb**: Vector database for embeddings
- **google-genai**: Google Gemini API client
- **pypdf**: PDF text extraction
- **scikit-learn**: K-Means clustering
- **numpy**: Numerical operations
- **tqdm**: Progress bars

## Troubleshooting

### SSL/OpenSSL Warning

If you see urllib3 SSL warnings, ensure you're using the project's virtual environment:
```bash
uv sync  # Recreate environment
uv run python cluster_pdfs.py  # Always use 'uv run'
```

### Healthcheck Failed

When using `chroma browse`, always specify the `--path` option:
```bash
uv run chroma browse pdf_cluster_data --local --path ./chroma_db
```

### No PDFs Found

Ensure PDF files are directly in the `./pdfs` directory (not in subdirectories)

### API Rate Limit (429 Error)

If you get "RESOURCE_EXHAUSTED" errors:
- Wait a few moments before retrying
- Check your [Google AI Studio quota](https://aistudio.google.com/)
- Consider using a different API key or upgrading your quota

### Out-of-Scope Questions

If the RAG system returns "no relevant information":
- Your question may be outside the scope of your PDF documents
- Try rephrasing with terms from your documents
- Adjust `RELEVANCE_THRESHOLD` in `rag_chat.py` (lower = more permissive)

## License

MIT
