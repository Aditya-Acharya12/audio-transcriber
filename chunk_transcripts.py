import nltk
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import hashlib

# Setup
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["audio_transcriber"]
source_collection = db["transcripts"]
chunk_collection = db["chunks"]

# Download NLTK data
nltk.download("punkt")

def generate_chunk_id(text, file_name, index):
    base = f"{file_name}_{index}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    return base

def chunk_text_with_overlap(text, max_words=250, overlap=50):
    words = word_tokenize(text)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        start += max_words - overlap

    return chunks

def process_transcripts():
    print("Starting to process transcripts...\n")
    docs = list(source_collection.find())

    if not docs:
        print("No documents found in 'transcripts' collection.")
        return

    for doc in docs:
        file_name = doc.get("file_name")
        text = doc.get("transcription")

        if not text or len(text.strip().split()) < 10:
            print(f"Skipping {file_name} — transcription is too short.")
            continue

        print(f"Found transcript for file: {file_name}")
        chunks = chunk_text_with_overlap(text)

        print(f"Chunked into {len(chunks)} overlapping segments.")
        inserted = 0

        for idx, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(chunk, file_name, idx)

            # Avoid duplicates
            if chunk_collection.find_one({"chunk_id": chunk_id}):
                continue

            chunk_doc = {
                "chunk_id": chunk_id,
                "file_name": file_name,
                "text": chunk,
                "embedding": None
            }
            chunk_collection.insert_one(chunk_doc)
            inserted += 1

        print(f"Inserted {inserted} chunks for {file_name}.\n")

    print("Done processing all transcripts.")

if __name__ == "__main__":
    process_transcripts()
