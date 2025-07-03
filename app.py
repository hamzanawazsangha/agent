import os
import json
import faiss
import numpy as np
import tempfile
import streamlit as st
import torch
import torchaudio
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
import whisper

# Initialize OpenAI and Whisper
client = OpenAI()
stt = whisper.load_model("base", device="cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load and chunk company data
with open('arslanasghar_full_content.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

chunk_size = 500
chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
embeddings = model.encode(chunks)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Create memory and log folders
os.makedirs("memory", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def retrieve_context(query, threshold=0.5, top_k=3):
    query_emb = model.encode([query])
    scores, indices = index.search(query_emb, top_k)
    if scores[0][0] >= threshold:
        return [chunks[i] for i in indices[0]]
    return []

def load_memory(user_id):
    path = f"memory/user_{user_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(user_id, messages):
    path = f"memory/user_{user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def append_log(user_id, user_text, ai_reply, intent=None):
    log_path = f"logs/{user_id}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\nIntent: {intent}\nArslan: {ai_reply}\n{'-'*50}\n")

def detect_intent(user_text):
    keywords = {
        "web": ["website", "web development", "site"],
        "seo": ["seo", "search engine"],
        "ads": ["ad", "google ads", "facebook ads", "campaign"],
        "pricing": ["price", "cost", "rate"],
        "support": ["help", "support", "issue"],
    }
    for intent, keys in keywords.items():
        if any(word in user_text.lower() for word in keys):
            return intent
    return "general"

# Streamlit UI
st.set_page_config(page_title="Arslan - Voice Assistant", layout="centered")
st.title("🎙️ Arslan — Your Human Digital Consultant")
st.markdown("Upload your voice message or type below. Arslan understands, remembers, and replies in your language.")

user_id = st.text_input("🔐 Enter your User ID:")
audio_bytes = st.file_uploader("🎧 Upload a voice message (WAV only):", type=["wav"])

typed_text = None
if user_id and not audio_bytes:
    typed_text = st.text_input("📝 Or type your question:")

if user_id and (audio_bytes or typed_text):
    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes.read())
            audio_path = tmp.name

        # ----- Improved Transcription Using Torchaudio -----
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16 kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Normalize to [-1, 1]
        waveform = waveform.squeeze().float()
        max_val = max(abs(waveform.max()), abs(waveform.min()))
        if max_val > 0:
            waveform = waveform / max_val

        # Pad or trim to fit Whisper's input size
        audio_tensor = whisper.pad_or_trim(waveform)
        mel = whisper.log_mel_spectrogram(audio_tensor).to(stt.device)

        # Decode
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(stt, mel, options)
        user_text = result.text.strip()

        if not user_text:
            st.warning("Sorry, couldn't understand the audio. Try again.")
            st.stop()
    else:
        user_text = typed_text.strip()

    language = detect(user_text)
    memory = load_memory(user_id)
    context = retrieve_context(user_text)
    context_text = "\n".join(context) if context else ""
    intent = detect_intent(user_text)

    discount_offer = ""
    if intent == "pricing":
        discount_offer = "You're eligible for a special 10% discount on all services today!"

    if not memory:
        memory.append({"role": "assistant", "content": "Hi there! I'm Arslan. May I know your name?"})
    elif len(memory) == 1:
        memory.append({"role": "assistant", "content": "Nice to meet you! How can I help you today?"})

    memory.append({"role": "user", "content": user_text})

    system_prompt = (
        f"You are Arslan, a friendly and human digital marketer from Doha, Qatar. "
        f"You are not an AI. Keep responses short and professional. Act like a real person and base your answers on the provided data. "
        f"Respond in {language.upper()} language."
    )
    if context_text:
        system_prompt += f"\nRelevant info: {context_text}"
    if discount_offer:
        memory.append({"role": "assistant", "content": discount_offer})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + memory[-10:]
    )
    ai_reply = response.choices[0].message.content.strip()

    memory.append({"role": "assistant", "content": ai_reply})
    save_memory(user_id, memory)
    append_log(user_id, user_text, ai_reply, intent)

    st.markdown(f"**🗣️ You said:** {user_text}")
    st.markdown(f"**🤖 Arslan replied:** {ai_reply}")

    with open("response.mp3", "wb") as f:
        audio_data = client.audio.speech.create(
            model="tts-1",
            voice="echo",  # Male-sounding voice
            input=ai_reply
        )
        f.write(audio_data.content)

    st.audio("response.mp3", format="audio/mp3")

elif user_id:
    st.info("Please upload your voice message or type your question to begin.")
