# gradio_app.py
# Gradio UI for AI Doctor

import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purpose. 
With what I see, I think you have .... If you make a differential, suggest some remedies for them. 
Your response should be in one long paragraph. Keep your answer concise (max 2 sentences)."""

def process_inputs(audio_filepath, image_filepath):
    # Step 1: Transcribe patient audio
    speech_to_text_output = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=audio_filepath,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    # Step 2: Analyze image + text
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + (speech_to_text_output or ""),
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # Step 3: Convert doctor response to speech
    voice_file = text_to_speech_with_elevenlabs(
        input_text=doctor_response,
        output_filepath="final.mp3"
    )

    return speech_to_text_output, doctor_response, voice_file

# Gradio UI
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Patient Voice"),
        gr.Image(type="filepath", label="Upload Image")
    ],
    outputs=[
        gr.Textbox(label="Patient Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice", autoplay=True)  # ✅ added autoplay
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)
