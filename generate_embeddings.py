from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client["audio_transcriber"]
chunk_collection = db["chunks"]

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings():
    print("Generating embeddings for unprocessed chunks...\n")
    count = 0

    for doc in chunk_collection.find({"embedding": None}):
        text = doc["text"]
        chunk_id = doc["chunk_id"]

        # Generate embedding
        embedding = model.encode(text).tolist()  # convert numpy to list

        # Update MongoDB document
        chunk_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": embedding}}
        )

        print(f"Embedded chunk: {chunk_id}")
        count += 1

    if count == 0:
        print("All chunks already have embeddings.")
    else:
        print(f"\nDone! Embedded {count} chunks.")

if __name__ == "__main__":
    generate_embeddings()
