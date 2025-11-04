def chunk_documents(docs,chunk_size=300,overlap=30):
    chunks = []
    for doc in docs:
        words = doc.split()
        for i in range (0,len(words),chunk_size-overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks