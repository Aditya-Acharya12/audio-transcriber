import nltk
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["audio_transcriber"]
source_collection = db["transcripts"]     # existing transcriptions
chunk_collection = db["chunks"]           # target collection for chunks

def chunk_text(text, max_words=250):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count <= max_words:
            current_chunk += " " + sentence
            current_length += word_count
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = word_count
    if current_chunk:
        chunks.append(current_chunk.strip())

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

        if not text:
            continue

        print(f"Found transcript for file: {file_name}")

        if len(text.split()) < 5:
            print("Skipping — transcription is too short.\n")
            continue

        chunks = chunk_text(text)
        print(f"Chunked into {len(chunks)} pieces.")

        inserted = 0
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_{idx}"

            # Check if chunk already exists
            if chunk_collection.find_one({"chunk_id": chunk_id}):
                continue  # skip if already exists

            chunk_doc = {
                "chunk_id": chunk_id,
                "file_name": file_name,
                "text": chunk,
                "embedding": None
            }
            chunk_collection.insert_one(chunk_doc)
            inserted += 1

        print(f"Inserted {inserted} new chunks for {file_name}.\n")

    print("Done processing all transcripts.")

if __name__ == "__main__":
    nltk.download('punkt')  # ensure tokenizer is available
    process_transcripts()