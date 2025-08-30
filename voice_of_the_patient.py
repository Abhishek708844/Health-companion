# voice_of_the_patient.py
# Handles patient voice recording + transcription with Groq

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

# FFmpeg path for Pydub (adjust if different)
AudioSegment.converter = r"C:\Users\95283\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Record audio from mic and save to mp3
def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            # Convert to MP3
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")

            return True

    except Exception as e:
        logging.error(f"Error during recording: {e}")
        return False

# Transcribe audio with Groq
def transcribe_with_groq(stt_model, audio_filepath, groq_api_key):
    try:
        client = Groq(api_key=groq_api_key)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None
