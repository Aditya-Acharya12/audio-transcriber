from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["audio_transcriber"]
chunk_collection = db["chunks"]

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_chunks(query, top_k=3):
    query_embedding = model.encode(query)

    similarities = []

    for doc in chunk_collection.find({"embedding": {"$ne": None}}):
        chunk_embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((similarity, doc))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop {top_k} relevant chunks for your query:\n")
    for i, (score, doc) in enumerate(similarities[:top_k]):
        print(f"Rank {i+1} | Score: {score:.4f}")
        print(f"File: {doc['file_name']}")
        print(f"Chunk ID: {doc['chunk_id']}")
        print(f"Text: {doc['text']}\n")

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    retrieve_top_chunks(user_query)