import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from load_docs import load_documents
from chunk import chunk_documents

DOCS_PATH = "../data/documents"
INDEX_PATH = "../data/vector_index"

def build_index():
    print("loading documents")
    docs= load_documents(DOCS_PATH)

    print("making chunks of documents")
    chunks = chunk_documents(docs)

    print("embedding the chunks")
    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    embeddings = model.encode(chunks)

    print("building Faiss index")
    dimentions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimentions)
    index.add(embeddings)

    os.makedirs(INDEX_PATH,exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_PATH,"faiss.index"))
    with open(os.path.join(INDEX_PATH,"chunks.pkl"),"wb") as f:
        pickle.dump(chunks,f)

    print ("index is built at:", INDEX_PATH)
if __name__=="__main__":
    build_index()