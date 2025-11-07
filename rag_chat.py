import os
from google import genai
from chromadb.utils import embedding_functions
import chromadb

# --- Configuration ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_cluster_data"
EMBEDDING_MODEL = "text-embedding-004"
GENERATIVE_MODEL = "gemini-2.0-flash-exp"
TOP_K_RESULTS = 5  # Number of relevant documents to retrieve
RELEVANCE_THRESHOLD = 0.3  # Minimum relevance score (0-1) to consider documents relevant

# Ensure your Google API Key is set
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set.")


class GoogleEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function for ChromaDB using Google's Gemini API."""
    
    def __init__(self):
        pass
    
    def __call__(self, input_data):
        client = genai.Client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=input_data,
            config={"taskType": "RETRIEVAL_QUERY"}
        )
        return [emb.values for emb in response.embeddings]


def initialize_chroma():
    """Initialize ChromaDB client and get collection."""
    print(f"🔌 Connecting to ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=GoogleEmbeddingFunction()
    )
    
    print(f"✅ Connected! Collection has {collection.count()} documents.")
    return collection


def retrieve_relevant_docs(collection: chromadb.Collection, query: str, n_results: int = TOP_K_RESULTS):
    """Query ChromaDB for relevant documents."""
    print(f"\n🔍 Searching for: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Filter results by relevance threshold
    if results['documents'][0]:
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            relevance_score = 1 - distance
            if relevance_score >= RELEVANCE_THRESHOLD:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
        
        results['documents'][0] = filtered_docs
        results['metadatas'][0] = filtered_metadatas
        results['distances'][0] = filtered_distances
    
    return results


def format_context(results):
    """Format retrieved documents into context for the LLM."""
    context_parts = []
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        cluster = metadata.get('cluster_id', 'N/A')
        
        context_parts.append(
            f"[Document {i+1}]\n"
            f"Source: {source} (Page {page}, Cluster {cluster})\n"
            f"Relevance Score: {1 - distance:.3f}\n"
            f"Content: {doc}\n"
        )
    
    return "\n---\n".join(context_parts)


def generate_response(query: str, context: str):
    """Generate AI response using Gemini with RAG context."""
    client = genai.Client()
    
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available. Always cite which documents you're referencing (e.g., "According to Document 1...").

ANSWER:"""

    print("\n🤖 Generating AI response...")
    
    try:
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "⚠️ API rate limit exceeded. Please wait a moment and try again."
        elif "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please check your Google AI Studio quota limits."
        else:
            raise


def chat_with_pdfs(collection: chromadb.Collection):
    """Interactive chat loop for querying PDFs."""
    print("\n" + "="*60)
    print("💬 RAG Chat System - Ask questions about your PDFs")
    print("="*60)
    print("Type 'quit' or 'exit' to end the conversation\n")
    
    while True:
        # Get user query
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            # Retrieve relevant documents
            results = retrieve_relevant_docs(collection, query)
            
            # Check if we got results
            if not results['documents'][0]:
                print("\n📭 I don't have any relevant information in my database to answer this question.")
                print("💡 This question appears to be outside the scope of the loaded documents.\n")
                continue
            
            # Format context
            context = format_context(results)
            
            # Generate response
            response = generate_response(query, context)
            
            # Display response
            print(f"\n🤖 Assistant: {response}\n")
            
            # Optionally show sources
            print("📚 Sources:")
            for i, metadata in enumerate(results['metadatas'][0][:3]):
                print(f"  {i+1}. {metadata.get('source')} (Page {metadata.get('page')})")
            print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def query_once(collection: chromadb.Collection, query: str):
    """Single query mode for scripting."""
    try:
        # Retrieve relevant documents
        results = retrieve_relevant_docs(collection, query)
        
        if not results['documents'][0]:
            print("\n📭 I don't have any relevant information in my database to answer this question.")
            print("💡 This question appears to be outside the scope of the loaded documents.")
            return
        
        # Format context
        context = format_context(results)
        
        # Generate response
        response = generate_response(query, context)
        
        # Display response
        print(f"\n🤖 Answer:\n{response}\n")
        
        # Show sources
        print("📚 Sources:")
        for i, metadata in enumerate(results['metadatas'][0]):
            print(f"  {i+1}. {metadata.get('source')} (Page {metadata.get('page')})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Initialize ChromaDB
    collection = initialize_chroma()
    
    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        query_once(collection, query)
    else:
        # Interactive mode
        chat_with_pdfs(collection)
