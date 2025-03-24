from fastapi import FastAPI
from pydantic import BaseModel
from rag import query_rag

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
def query(request: QueryRequest):
    return query_rag(request.query)