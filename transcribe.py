import boto3  # AWS SDK for Python
import whisper
import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, UTC
import time
from transformers import pipeline

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["audio_transcriber"] 
collection = db["transcripts"]

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def save_to_db(file_name, language, duration, transcription):
    doc = {
        "file_name": file_name,
        "language": language,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration": duration,
        "transcription": transcription
    }
    collection.insert_one(doc)

def download_from_s3(bucket_name, audio_key, local_audio):
    s3 = boto3.client('s3')   # creates an s3 client
    s3.download_file(bucket_name,audio_key, local_audio)
    print("Audio file downloaded from S3")

def transcribe_audio(local_audio, local_transcript):
    model = whisper.load_model("base")
    result = model.transcribe(local_audio)

    transcribed_text = result["text"]
    language = result["language"]
    duration = result["segments"][-1]["end"] if result["segments"] else 0

    with open(local_transcript, "w") as f:
        f.write(transcribed_text)
    save_to_db(local_audio, language, duration, transcribed_text)
    print("Transcription complete and saved locally and stored in DB")

def is_already_transcribed(file_name):
    return collection.find_one({"file_name": file_name}) is not None

def upload_to_s3(bucket_name, local_transcript, transcript_key):
    s3 = boto3.client("s3")
    s3.upload_file(local_transcript, bucket_name, transcript_key)
    print("Transcription uploaded to S3")

def summarize_transcript(local_transcript, max_input_chars=3000):
    with open(local_transcript, 'r') as f:
        text = f.read()
    if len(text) > max_input_chars:
        text = text[:max_input_chars]
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file from S3 using whisper")
    parser.add_argument("--bucket", type=str,help="S3 bucket name")
    parser.add_argument("--local_file",type=str,help="Path to loacl audio file")
    parser.add_argument("--audio_key", type=str, help="S3 audio file key")
    parser.add_argument("--upload", action="store_true",help="Upload transcript to S3")

    args = parser.parse_args()

    bucket_name = args.bucket
    if args.local_file and (args.bucket or args.audio_key):
        print("Error: Please provide either --local_file OR --bucket and --audio_key, not both.")
        return

    if args.local_file:
        local_audio = args.local_file
        filename = os.path.basename(local_audio)
        local_transcript = os.path.splitext(filename)[0] + ".txt"
        transcript_key = "transcripts/" + local_transcript

    elif args.audio_key and args.bucket:
        audio_key = args.audio_key
        local_audio = os.path.split(audio_key)[1]
        local_transcript = os.path.splitext(local_audio)[0] + ".txt"
        transcript_key = "transcripts/" + local_transcript

        start = time.time()
        download_from_s3(bucket_name, audio_key, local_audio)
        print(f"Downloaded audio file in {time.time() - start:.2f} seconds")

    else:
        print("ERROR: Either --local_file or --audio_key and --bucket must be provided.")
        return

    if is_already_transcribed(local_audio):
        print(f"Skipping transcription for {local_audio} as it already exists in the database.")
    else:
        start = time.time()
        transcribe_audio(local_audio,local_transcript)
        print(f"Transcription completed in {time.time() - start:.2f} seconds")
        start = time.time()
        summary = summarize_transcript(local_transcript)
        print(f"Summarization completed in {time.time() - start:.2f} seconds")
        print(summary)

        if args.upload:
            if not args.bucket:
                print("ERROR: --upload requires --bucket to be specified.")
                return
            start = time.time()
            upload_to_s3(bucket_name, local_transcript, transcript_key)
            print(f"Uploaded transcription in {time.time() - start:.2f} seconds")
if __name__ == "__main__":
    main()