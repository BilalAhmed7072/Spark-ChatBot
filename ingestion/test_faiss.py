import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
INDEX_PATH = "../data/vector_index"
FAISS_FILE = os.path.join(INDEX_PATH, "faiss.index")
CHUNKS_FILE = os.path.join(INDEX_PATH, "chunks.pkl")
print("🔹 Loading FAISS index and chunks...")
index = faiss.read_index(FAISS_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)
print("🔹 Loading embedding model...")
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
query = " what are seo strategies of spark solutionz?"
print(f"\n🔍 Query: {query}")
query_embedding = model.encode([query])
k = 1
D, I = index.search(query_embedding, k)

print("\n📄 Top results:")
for idx, distance in zip(I[0], D[0]):
    print(f"\n--- Result ---")
    print(f"Chunk index: {idx}")
    print(f"Score: {distance:.4f}")
    print(f"Text:\n{chunks[idx][:500]}...") 
import gc
import faiss
import torch

faiss.index_factory(1, "Flat")  # dummy call to safely unload
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

