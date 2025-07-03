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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue
import uuid

# Initialize OpenAI and models on CPU
client = OpenAI()
stt = whisper.load_model("base", device="cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load and chunk company data
with open('arslanasghar_full_content.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

chunk_size = 500
chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
embeddings = model.encode(chunks)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Setup directories
os.makedirs("memory", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Context Retrieval
def retrieve_context(query, threshold=0.5, top_k=3):
    query_emb = model.encode([query])
    scores, indices = index.search(query_emb, top_k)
    if scores[0][0] >= threshold:
        return [chunks[i] for i in indices[0]]
    return []

# Memory Handling
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

# Log Chat History
def append_log(user_id, user_text, ai_reply, intent=None):
    log_path = f"logs/{user_id}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\nIntent: {intent}\nArslan: {ai_reply}\n{'-' * 50}\n")

# Intent Detection
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

# WebRTC Audio Capture
audio_queue = queue.Queue()
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_queue.put(frame.to_ndarray().flatten())
        return frame

# Streamlit UI
st.set_page_config(page_title="Arslan - Voice Assistant", layout="centered")
st.title("🎙️ Arslan — Your Human Digital Consultant")
st.markdown("Use your voice to talk. Arslan understands, remembers, and replies smartly.")

user_id = st.text_input("🔐 Enter your User ID:")
audio_bytes = st.file_uploader("🎧 Upload a voice message (.wav only):", type=["wav"])

st.markdown("---")
st.markdown("🎤 **Or use your microphone below (Chrome/Edge recommended):**")

webrtc_ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Process Input
if user_id and (audio_bytes or (webrtc_ctx and webrtc_ctx.state.playing and not audio_bytes)):
    with st.spinner("🔄 Processing your voice..."):
        if audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes.read())
                audio_path = tmp.name
        else:
            st.warning("Recording in progress... please speak clearly.")
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            if not audio_data:
                st.stop()
            audio_array = np.concatenate(audio_data).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, samplerate=16000)
                audio_path = tmp.name

        result = stt.transcribe(audio_path)
        user_text = result["text"].strip()
        language = detect(user_text)
        memory = load_memory(user_id)
        context = retrieve_context(user_text)
        intent = detect_intent(user_text)

        if not memory:
            memory.append({"role": "assistant", "content": "Hi there! I'm Arslan. May I know your name?"})
        elif len(memory) == 1:
            memory.append({"role": "assistant", "content": "Nice to meet you! How can I help you today?"})

        memory.append({"role": "user", "content": user_text})

        discount_offer = "You're eligible for a special 10% discount on all services today!" if intent == "pricing" else ""

        system_prompt = (
            f"You are Arslan, a friendly and human digital marketer from Doha, Qatar. "
            f"Your task is to handle the user's query professionally in {language.upper()} without acting like an AI."
        )
        if context:
            system_prompt += f"\nRelevant info: {' '.join(context)}"
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

        st.success("✅ Response generated!")

        st.markdown(f"**🗣️ You said:** {user_text}")
        st.markdown(f"**🤖 Arslan replied:** {ai_reply}")

        response_filename = f"response_{uuid.uuid4().hex}.mp3"
        with open(response_filename, "wb") as f:
            audio_data = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=ai_reply
            )
            f.write(audio_data.content)

        st.audio(response_filename, format="audio/mp3")

elif user_id:
    st.info("Please upload a `.wav` file or allow mic access to start speaking.")
