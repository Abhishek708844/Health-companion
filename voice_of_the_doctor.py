# voice_of_the_doctor.py
# Handles doctor text-to-speech (gTTS + ElevenLabs)

import os
from gtts import gTTS
from elevenlabs import save
from elevenlabs.client import ElevenLabs

# API key for ElevenLabs
ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

# ---------------------------
# gTTS fallback
# ---------------------------
def text_to_speech_with_gtts(input_text, output_filepath):
    try:
        tts = gTTS(text=input_text, lang="en", slow=False)
        tts.save(output_filepath)
        return output_filepath
    except Exception as e:
        print(f"❌ gTTS failed: {e}")
        return None

# ---------------------------
# ElevenLabs TTS
# ---------------------------
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    if not ELEVENLABS_API_KEY:
        print("⚠️ No ELEVEN_API_KEY found, using gTTS fallback.")
        return text_to_speech_with_gtts(input_text, output_filepath)

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.generate(
            text=input_text,
            voice="Aria",  # change voice if needed
            model="eleven_turbo_v2"
        )
        save(audio, output_filepath)
        return output_filepath
    except Exception as e:
        print(f"❌ ElevenLabs failed: {e}, falling back to gTTS")
        return text_to_speech_with_gtts(input_text, output_filepath)
