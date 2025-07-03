import os
import json
import faiss
import numpy as np
import tempfile
import soundfile as sf
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
import whisper
from io import BytesIO

# ------------------ Cached Initializations ------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_data
def load_and_chunk_data(path="arslanasghar_full_content.txt", chunk_size=500):
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    return chunks, idx

embedder = load_embedding_model()
stt = load_whisper_model()
client = OpenAI()

chunks, index = load_and_chunk_data()

# Ensure folders exist
os.makedirs("memory", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ------------------ Helper Functions ------------------

def retrieve_context(query, threshold=0.5, top_k=3):
    query_emb = embedder.encode([query])
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

# ------------------ Streamlit UI ------------------

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
        result = stt.transcribe(audio_path)
        user_text = result["text"].strip()
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

    if intent == "pricing":
        memory.append({"role": "assistant", "content": "You're eligible for a special 10% discount on all services today!"})

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

    audio_data = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=ai_reply
    )
    audio_bytes = BytesIO(audio_data.content)
    st.audio(audio_bytes, format="audio/mp3")

elif user_id:
    st.info("Please upload your voice message or type your question to begin.")
