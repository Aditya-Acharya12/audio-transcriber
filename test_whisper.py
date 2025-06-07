import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")
AudioFile = "test.mp3"

#transcribe the audio file 
result = model.transcribe(AudioFile, language="en")
transcribed_text = result["text"]

base_name = os.path.splitext(AudioFile)[0]
output_file = base_name + ".txt"
with open(output_file, "w") as f:
    f.write(transcribed_text)