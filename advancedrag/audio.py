import io
import re
import streamlit as st
from gtts import gTTS


def generate_audio_from_text(text):
    try:
        clean_text = re.sub(r'<.*?>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        match = re.split(r'\n\n<span style=', clean_text)
        if match and len(match) > 0:
            clean_text = match[0]
        audio_buffer = io.BytesIO()
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None


def clean_text_for_audio(text):
    clean_text = re.sub(r'<.*?>', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if "Source:" in clean_text:
        clean_text = clean_text.split("Source:")[0].strip()
    return clean_text