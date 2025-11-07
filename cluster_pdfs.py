import os
import glob
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from google import genai

# --- Configuration ---
PDF_DIR = "./pdfs"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "text-embedding-004" # Recommended Google Embedding Model
K_CLUSTERS = 10 # Number of clusters for K-Means

# Ensure your Google API Key is set as an environment variable
# export GEMINI_API_KEY='YOUR_API_KEY'
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- 1. PDF Text Extraction ---
def extract_text_from_pdfs(pdf_dir: str) -> list:
    """Extracts text from all PDF files in a directory."""
    documents = []
    print(f"🚀 Starting PDF extraction from {pdf_dir}...")
    
    # Use glob to find all PDF files recursively
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"), recursive=True)
    
    for file_path in pdf_files:
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Store as a simple dictionary
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1 
                        }
                    })
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            continue
            
    print(f"✅ Extracted text from {len(pdf_files)} files into {len(documents)} pages.")
    return documents

# --- 2. Chroma Setup and Embedding ---

def create_chroma_collection(db_path: str, name: str, docs: list):
    """Initializes ChromaDB, embeds documents, and stores them."""
    
    # 2.1 Custom Google Embedding Function for Chroma
    class GoogleEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, input_data):
            client = genai.Client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=input_data,
                config={"taskType": "RETRIEVAL_DOCUMENT"}
            )
            # response.embeddings is a list of Embedding objects
            return [emb.values for emb in response.embeddings]

    # 2.2 Initialize Chroma Client
    print(f"🛠️ Initializing Chroma client at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # 2.3 Create/Get Collection with the Google Embedding Function
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=GoogleEmbeddingFunction()
    )
    
    # 2.4 Prepare Data for Chroma (IDs must be unique)
    ids = [f"{doc['metadata']['source']}_p{doc['metadata']['page']}" for doc in docs]
    texts = [doc['text'] for doc in docs]
    metadatas = [doc['metadata'] for doc in docs]
    
    print(f"📤 Adding {len(ids)} document chunks to Chroma...")
    # Chroma handles chunking internally if documents are large, but here we 
    # treat each page as a 'chunk'. 
    # For better RAG, a more sophisticated TextSplitter (e.g., LangChain's) 
    # would be used before this step.
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"🎉 Documents added. Total count: {collection.count()}")
    return collection

# --- 3. Clustering ---

def perform_clustering(collection: chromadb.Collection):
    """Retrieves vectors, performs K-Means clustering, and updates Chroma metadata."""
    
    print(f"\n🧠 Starting K-Means Clustering with K={K_CLUSTERS}...")
    
    # Retrieve all data including IDs and Embeddings
    results = collection.get(
        ids=collection.get()['ids'], # Get all IDs
        include=["embeddings"]
    )
    
    # Convert embeddings to NumPy array for scikit-learn
    vectors = np.array(results['embeddings'])
    
    # Adjust K if we have fewer data points than requested clusters
    k = min(K_CLUSTERS, len(vectors))
    
    if len(vectors) < 2:
        print(f"🚫 Not enough data ({len(vectors)} points) to perform clustering. Need at least 2 points.")
        return
    
    if k < K_CLUSTERS:
        print(f"⚠️  Adjusting K from {K_CLUSTERS} to {k} due to limited data ({len(vectors)} points).")

    # Perform K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    print(f"📊 Clustering complete. Updating Chroma metadata...")
    
    # Prepare data for updating Chroma
    ids_to_update = results['ids']
    # Chroma update requires documents/embeddings/metadatas in the update call, 
    # but we only need to update metadatas.
    
    for doc_id, label in tqdm(zip(ids_to_update, cluster_labels), total=len(ids_to_update), desc="Updating DB"):
        # The simplest way to update a field is to re-get the original metadata 
        # and then update it, since Chroma doesn't have a single-field update.
        # Here, we fetch the existing metadata for the ID:
        
        # A full metadata update requires retrieving the existing metadata first
        # For simplicity and performance, we'll assume the retrieved IDs 
        # match the order of the original documents (a risk, but common in simple scripts)
        
        # NOTE: A better way is to iterate over a full retrieval and update one by one.
        # Since the retrieved `results` only contains `ids` and `embeddings`, 
        # we must update the metadata explicitly.
        
        # For this script, we'll create the new metadata list and update all at once
        # (This avoids hitting the DB repeatedly in the loop)
        
        # 1. Fetch ALL existing metadatas (since we only retrieved embeddings earlier)
        full_results = collection.get(ids=ids_to_update, include=["metadatas"])
        
        # 2. Add the cluster_id to the metadata
        updated_metadatas = []
        for i, metadata in enumerate(full_results['metadatas']):
            metadata['cluster_id'] = int(cluster_labels[i])
            updated_metadatas.append(metadata)

        # 3. Perform the bulk update 
        collection.update(
            ids=ids_to_update,
            metadatas=updated_metadatas
        )

    print("✅ Chroma metadata updated with cluster_id.")
    
    # Display results summary
    print("\n--- Clustering Summary ---")
    for i in range(k):
        count = sum(cluster_labels == i)
        print(f"Cluster {i:02d}: {count} documents ({(count/len(vectors))*100:.2f}%)")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Extract Text
    all_docs = extract_text_from_pdfs(PDF_DIR)
    
    if not all_docs:
        print("🛑 No text extracted. Check PDF_DIR and file integrity.")
    else:
        # 2. Create Chroma Collection (Embeddings happen here)
        # NOTE: Chroma will automatically use the Google embedding function
        # to generate and store the vectors for the documents.
        chroma_collection = create_chroma_collection(CHROMA_PATH, COLLECTION_NAME, all_docs)
        
        # 3. Perform Clustering and Update DB
        perform_clustering(chroma_collection)
