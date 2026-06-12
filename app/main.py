from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from app.models import QueryRequest, QueryResponse
from app.rag_logic import process_rag_query


app = FastAPI(
    title="RAG API", 
    description="backend for CV NLP processing.",
)

# / refers to the root address
@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url="/docs")

@app.post("/api/query", response_model=QueryResponse)
def query_candidates(request: QueryRequest):
    try:
        answer, chunks = process_rag_query(request.query, request.top_k)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"interal server error: {str(e)}")