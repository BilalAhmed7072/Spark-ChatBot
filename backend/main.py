from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from retrieval import retriever
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the persistent ChromaDB retriever safely before the app starts.
    This replaces the deprecated @app.on_event("startup").
    """
    try:
        retriever.initialize()  
        print("RAG retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise e
    yield
    print("Shutting down FastAPI app...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    """
    Endpoint for querying the RAG chatbot.
    """
    try:
        answer = retriever.answer_with_context(req.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8048,
        workers=1,
        loop="asyncio",
        reload=False
    )
