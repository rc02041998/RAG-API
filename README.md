# RAG-Based FastAPI Service with FAISS and Ollama

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI as the backend framework. It leverages FAISS for efficient similarity search and Ollama's Llama 3.2 model for generating responses.

## Features
- Uses FAISS for storing and retrieving document embeddings
- Embeddings generated using Ollama's Llama 3.2 model
- FastAPI-based API for querying the system
- Automatic FAISS index persistence
- Preloaded documents for retrieval

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies (FAISS, Ollama, NumPy, FastAPI, etc.)

### Install Dependencies
```sh
pip install fastapi faiss-cpu numpy ollama uvicorn
```

## Running the Application
### Start the FastAPI Server
```sh
uvicorn main:app --reload
```
This will start the FastAPI server on `http://127.0.0.1:8000`.

## API Endpoints
### Query the RAG System
**Endpoint:** `POST /query/`

**Request Body:**
```json
{
    "query": "Your question here"
}
```

**Response:**
```json
{
    "response": "Generated answer based on retrieved documents."
}
```

## File Structure
```
.
├── main.py         # FastAPI application
├── rag.py          # FAISS and Ollama integration
├── faiss_index.bin # FAISS index (if exists)
├── README.md       # Documentation
```

## How It Works
1. **Embeddings Generation:**
   - Uses Ollama's `llama3.2` model to generate embeddings for documents and queries.
2. **FAISS Indexing:**
   - Documents are stored in FAISS for efficient similarity-based retrieval.
   - If a saved FAISS index exists, it is loaded; otherwise, a new one is created.
3. **Query Handling:**
   - The query is embedded and searched against the FAISS index.
   - The top retrieved documents are used as context for the LLM to generate a response.
4. **Cleanup and Persistence:**
   - FAISS index is saved to disk.
   - Garbage collection ensures memory efficiency.

## Enhancements & Future Improvements
- Add support for dynamic document ingestion.
- Implement better prompt engineering for LLM responses.
- Optimize FAISS for large-scale data.



