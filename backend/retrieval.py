import os
import pickle
from sentence_transformers import SentenceTransformer
from servicellm import generate_llm_response
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_index/chunks.pkl")
PERSIST_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
os.makedirs(PERSIST_PATH, exist_ok=True)

class RAGRetriever:
    """
    RAG retriever using ChromaDB v0.4+ with permanent persistence.
    Fully safe on macOS and avoids deprecated Chroma settings.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
        self.chunks = None
        self.initialized = False

    def initialize(self):
        """Initialize PersistentClient, load chunks, and embedding model."""
        if self.initialized:
            return

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

        self.client = PersistentClient(
            path=PERSIST_PATH,     
            settings=Settings()   
        )

        self.collection = self.client.get_or_create_collection(name="rag_docs")

        new_ids = [str(i) for i in range(len(self.chunks))]

        #existing_ids = set(self.collection.get(include=["ids"])["ids"])
        #new_ids = [str(i) for i in range(len(self.chunks)) if str(i) not in existing_ids]

        if new_ids:
            new_docs = [self.chunks[int(i)] for i in new_ids]
            embeddings = self.model.encode(new_docs).tolist()
            self.collection.add(
                ids=new_ids,
                documents=new_docs,
                embeddings=embeddings
            )

        self.initialized = True
        print(f"RAG retriever initialized with Persistent ChromaDB at {PERSIST_PATH}.")

    def retrieve_query(self, query: str, top_k: int = 5):
        """Retrieve top-k most relevant document chunks using ChromaDB."""
        if not self.initialized:
            raise RuntimeError("RAGRetriever not initialized.")

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents"]
        )
        return results['documents'][0]

    def answer_with_context(self, query: str, top_k: int = 5):
        """Retrieve chunks and generate LLM answer using Groq."""
        context_chunks = self.retrieve_query(query, top_k)
        context_text = "\n\n".join(context_chunks)

        prompt = f"""You are an intelligent assistant. Use the following retrieved context to answer the user's question.
If the context does not contain the answer, say 'I could not find a clear answer in the provided data.'

Context:
{context_text}

Question: {query}

Answer:
"""
        response = generate_llm_response(prompt)
        return response
retriever = RAGRetriever()
