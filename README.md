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

## Configuration

Edit the constants in `cluster_pdfs.py`:

```python
PDF_DIR = "./pdfs"              # PDF input directory
CHROMA_PATH = "./chroma_db"     # ChromaDB storage path
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "text-embedding-004"
K_CLUSTERS = 10                 # Number of clusters
```

## Project Structure

```
pdf-to-vector/
├── pdfs/                  # Place your PDF files here
├── chroma_db/            # ChromaDB persistent storage
├── cluster_pdfs.py       # Main clustering script
├── main.py               # Entry point script
├── browse.sh             # Helper script to browse DB
├── Makefile              # Task automation
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

## How It Works

1. **Text Extraction**: PyPDF extracts text from each page of every PDF in the `pdfs/` directory

2. **Embedding Generation**: Each page's text is sent to Google's Gemini API to generate a vector embedding using the `text-embedding-004` model

3. **Storage**: Documents, embeddings, and metadata are stored in ChromaDB with unique IDs formatted as `{filename}_p{page_number}`

4. **Clustering**: K-Means algorithm groups similar documents based on their vector embeddings

5. **Metadata Update**: Each document is tagged with its `cluster_id` in ChromaDB

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

## License

MIT
