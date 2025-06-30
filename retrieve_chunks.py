from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["audio_transcriber"]
chunk_collection = db["chunks"]

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_and_rerank(query, top_k=3):
    print(f"\nQuery: {query}\n")

    # Embed the query
    query_embedding = model.encode(query)

    # Retrieve all chunks with embeddings
    chunks = list(chunk_collection.find({"embedding": {"$ne": None}}))

    if not chunks:
        print("No chunks with embeddings found.")
        return

    # Compute cosine similarity between query and each chunk
    scored_chunks = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk["embedding"], dtype=np.float32)
        score = util.cos_sim(
            torch.tensor(query_embedding, dtype=torch.float32),
            torch.tensor(chunk_embedding, dtype=torch.float32)
        ).item()
        scored_chunks.append((score, chunk))

    # Sort and get top-k results
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored_chunks[:top_k]

    # Display results
    for rank, (score, chunk) in enumerate(top_chunks, 1):
        print(f"Rank {rank} | Score: {score:.4f}")
        print(f"File: {chunk['file_name']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Text: {chunk['text']}\n")

if __name__ == "__main__":
    query = input("Enter your question: ")
    retrieve_and_rerank(query)