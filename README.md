# 🎙️ Whisper S3 Transcriber

A Python script that:

- Downloads audio files from AWS S3 (or uses local audio)
- Transcribes them using OpenAI's Whisper
- Summarizes the transcript using a transformer model
- Stores transcriptions in a local SQLite database
- (Optionally) uploads transcriptions back to S3

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/whisper-s3-transcriber.git
cd whisper-s3-transcriber
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt`**
```
boto3
openai-whisper
transformers
torch
sentencepiece
```

### 3. Install `ffmpeg`

Whisper requires `ffmpeg` to be installed and available in your system's PATH.

#### macOS (Homebrew):
```bash
brew install ffmpeg
```

#### Ubuntu / Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows:
- Download from: https://ffmpeg.org/download.html
- Extract the files and add the `bin/` folder to your system’s PATH

To verify installation:

```bash
ffmpeg -version
```

---

## 🔐 AWS Credentials Setup

This script uses `boto3` to access AWS S3. You must configure your AWS credentials.

### Configure via CLI (Recommended):

```bash
aws configure
```

This creates the credentials file at `~/.aws/credentials`:

```
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

You can also set the region in `~/.aws/config`:

```
[default]
region = us-east-1
```

Alternatively, you can set environment variables:

```bash
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
```

---

## ▶️ How to Run the Script

Make sure you're in the project directory and your virtual environment is activated.

### Option 1: Transcribe a **local audio** file

```bash
python transcriber.py --local_file path/to/audio.mp3
```

### Option 2: Transcribe an **audio file from S3**

```bash
python transcriber.py --bucket your-bucket-name --audio_key path/in/bucket/audio.mp3
```

### Option 3: Transcribe from S3 and upload transcript back to S3

```bash
python transcriber.py --bucket your-bucket-name --audio_key path/in/bucket/audio.mp3 --upload
```

---

## 🧠 What Happens Under the Hood?

- Downloads the audio (if from S3)
- Skips transcription if already processed (based on filename)
- Transcribes the audio using Whisper
- Summarizes the first ~3000 characters of the transcript
- Saves the transcript to:
  - A `.txt` file
  - A local SQLite database (`transcriptions.db`)
- Uploads `.txt` to S3 if `--upload` is specified

---

## 🗃 Example Entry in `transcriptions.db`

| id | file_name     | language | timestamp                  | duration | transcription |
|----|---------------|----------|----------------------------|----------|----------------|
| 1  | audio.mp3     | en       | 2025-06-05T12:34:56+00:00  | 120.5    | ...text...     |

---


---

## 🙌 Acknowledgements

- [Whisper by OpenAI](https://github.com/openai/whisper)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
- [ffmpeg](https://ffmpeg.org/)
