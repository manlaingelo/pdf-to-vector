import chromadb
from chromadb.utils import embedding_functions
import requests
import json

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "all-minilm:l6-v2"
RELEVANCE_THRESHOLD = 0.3

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input_data):
        embeddings = []
        for text in input_data:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': EMBEDDING_MODEL, 'prompt': text},
            )
            response.raise_for_status()
            embeddings.append(json.loads(response.text)['embedding'])
        return embeddings

# Initialize
print("🔌 Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=OllamaEmbeddingFunction())

print(f"✅ Total documents in collection: {collection.count()}\n")

# Sample some documents
sample = collection.get(limit=3, include=['documents', 'metadatas'])
print("📄 Sample documents:")
for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
    print(f"\n  [{i+1}] {meta.get('source')} (Page {meta.get('page')})")
    print(f"      Content preview: {doc[:200]}...")

# Test query
query = "what is machine learning"
print(f"\n\n🔍 Testing query: '{query}'")

results = collection.query(
    query_texts=[query],
    n_results=10,
    include=['documents', 'metadatas', 'distances']
)

print(f"\n📊 Raw results: {len(results['documents'][0])} documents found")

if results['documents'][0]:
    print(f"\n📈 Distances and Relevance Scores:")
    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0], 
        results['distances'][0]
    )):
        relevance = 1 - distance
        passed = "✅" if relevance >= RELEVANCE_THRESHOLD else "❌"
        print(f"  {passed} [{i+1}] Distance: {distance:.4f}, Relevance: {relevance:.4f}")
        print(f"      Source: {meta.get('source')} (Page {meta.get('page')})")
        print(f"      Content: {doc[:150]}...\n")
else:
    print("❌ No results returned from query!")

print(f"\n⚙️  Current RELEVANCE_THRESHOLD: {RELEVANCE_THRESHOLD}")
print(f"    (Documents with relevance < {RELEVANCE_THRESHOLD} are filtered out)")
