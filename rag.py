import faiss
import numpy as np
import ollama
import gc
import atexit
import os

# Set FAISS to use a single thread to prevent multiprocessing issues
faiss.omp_set_num_threads(1)

# Initialize Ollama with Llama 3.2 model
model_name = "llama3.2"

# FAISS index file path
faiss_index_path = "faiss_index.bin"

# Document storage (Preloaded texts)
document_texts = ["Preloaded document 1", "Preloaded document 2"]

def get_llama_embedding(text: str):
    """Generate embeddings using Llama 3.2 via Ollama."""
    response = ollama.embeddings(model=model_name, prompt=text)
    return np.array(response['embedding'], dtype=np.float32)

# Load FAISS index or create a new one
if os.path.exists(faiss_index_path):
    try:
        index = faiss.read_index(faiss_index_path)
        print(f"FAISS index loaded successfully with {index.ntotal} vectors.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        index = None
else:
    print("FAISS index file not found. Creating a new index.")
    sample_embedding = get_llama_embedding("test text")  # Generate one embedding to get the dimension
    embedding_dim = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    faiss.write_index(index, faiss_index_path)

def add_documents_to_faiss(docs):
    """Add documents to the FAISS index."""
    global index, document_texts

    if not isinstance(docs, list):
        raise ValueError("Documents should be a list of strings.")

    embeddings = np.array([get_llama_embedding(doc) for doc in docs], dtype=np.float32)

    # Ensure FAISS index has the correct dimensions
    if index.d == 0 or index.d != embeddings.shape[1]:
        print(f"Reinitializing FAISS index with dimension: {embeddings.shape[1]}")
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    document_texts.extend(docs)
    faiss.write_index(index, faiss_index_path)
    print(f"Added {len(docs)} documents to FAISS index. Total documents: {len(document_texts)}")

def query_rag(query: str):
    """Retrieve documents and generate a response."""
    if index is None or index.ntotal == 0:
        return {"response": "FAISS index not available or empty."}
    
    query_embedding = get_llama_embedding(query).reshape(1, -1)
    
    # Ensure query embedding matches FAISS index dimension
    if query_embedding.shape[1] != index.d:
        return {"response": f"Embedding dimension mismatch. Expected {index.d}, got {query_embedding.shape[1]}"}

    _, indices = index.search(query_embedding, k=3)  # Retrieve top 3 documents

    retrieved_texts = [document_texts[i] for i in indices[0] if i < len(document_texts)]
    context = "\n".join(retrieved_texts)
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return {"response": response['message']['content'].strip()}

# **Preload documents into FAISS at startup**
if index.ntotal == 0:
    add_documents_to_faiss(document_texts)

# Cleanup functions
def close_faiss():
    """Explicitly delete FAISS index object."""
    global index
    if index is not None:
        del index

def cleanup():
    """Perform garbage collection."""
    gc.collect()

# Register cleanup handlers
atexit.register(close_faiss)
atexit.register(cleanup)