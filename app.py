from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import requests
from openai import OpenAI
# from src.helper import (describe_image, download_hugging_face_embeddings)
# from src.prompt import chat_prompt, IMAGE_DESCRIPTION_PROMPT
from src.prompt import *
import re

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

from src.classifier import classify_intent
from src.prompt import chat_prompt, IMAGE_DESCRIPTION_PROMPT
from src.helper import (
    describe_image,
    download_hugging_face_embeddings,
    transcribe_audio
)


# =========================================================
# ENV
# =========================================================


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_MAPS_API_KEY"] = GOOGLE_MAPS_API_KEY
# =========================================================
# ENV (STRICT, REQUIRED)
# =========================================================

def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in GitHub Secrets and pass it to docker run."
        )
    return value

OPENAI_API_KEY = require_env("OPENAI_API_KEY")
PINECONE_API_KEY = require_env("PINECONE_API_KEY")
GOOGLE_MAPS_API_KEY = require_env("GOOGLE_MAPS_API_KEY")

# Only strings reach here — safe to assign
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_MAPS_API_KEY"] = GOOGLE_MAPS_API_KEY

# =========================================================
# APP
# =========================================================

app = Flask(__name__)

# =========================================================
# EMERGENCY DETECTION (SECONDARY SAFETY NET)
# =========================================================

def is_serious_medical(text: str) -> bool:
    text = text.lower()
    return any(flag in text for flag in [
        "chest pain", "shortness of breath", "can't breathe",
        "difficulty breathing", "stroke", "seizure",
        "unconscious", "passed out", "fainted",
        "severe bleeding", "overdose", "anaphylaxis",
        "suicidal", "kill myself"
    ])

def emergency_message() -> str:
    return (
        "[[EMERGENCY]]\n"
        "⚠️ **This may be a medical emergency.**\n\n"
        "Please call emergency services immediately.\n\n"
        "I can help find nearby hospitals."
    )

# =========================================================
# GOOGLE MAPS
# =========================================================

def get_nearby_hospitals(lat, lng):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 5000,
        "type": "hospital",
        "key": GOOGLE_MAPS_API_KEY
    }
    return requests.get(url, params=params).json().get("results", [])

def add_distances(lat, lng, hospitals):
    if not hospitals:
        return []

    destinations = "|".join(
        f"{h['geometry']['location']['lat']},{h['geometry']['location']['lng']}"
        for h in hospitals
    )

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": f"{lat},{lng}",
        "destinations": destinations,
        "mode": "driving",
        "key": GOOGLE_MAPS_API_KEY
    }

    data = requests.get(url, params=params).json()
    enriched = []

    for i, h in enumerate(hospitals):
        el = data["rows"][0]["elements"][i]
        if el["status"] == "OK":
            h["distance"] = el["distance"]["text"]
            h["duration"] = el["duration"]["text"]
            h["duration_value"] = el["duration"]["value"]
            enriched.append(h)

    return sorted(enriched, key=lambda x: x["duration_value"])[:5]

def format_hospitals(hospitals):
    if not hospitals:
        return "I couldn’t find nearby hospitals."

    lines = ["🏥 **Nearest hospitals to you:**\n"]
    for i, h in enumerate(hospitals, 1):
        lines.append(
            f"{i}️⃣ {h['name']}\n"
            f"📍 {h.get('vicinity')}\n"
            f"📏 {h['distance']} • ⏱ {h['duration']}\n"
        )
    return "\n".join(lines)

# =========================================================
# RAG SETUP
# =========================================================

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

base_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["question"])
    )
    | chat_prompt
    | chat_model
    | StrOutputParser()
)

chat_histories = {}

def get_chat_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("chat.html")

# -------------------------
# TEXT CHAT
# -------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    session_id = request.remote_addr

    intent = classify_intent(msg)

    # 🚨 Emergency
    if intent == "EMERGENCY" or is_serious_medical(msg):
        return emergency_message()

    # ✅ Medical only
    if intent in {"MEDICAL_QUESTION", "MEDICAL_FOLLOWUP"}:
        return rag_chain.invoke(
            {"question": msg},
            config={"configurable": {"session_id": session_id}}
        )

    # 👋 Greeting
    if intent == "GREETING":
        return "Hi! 👋 I can help with medical questions."

    # ❌ Out of scope
    return "I can only help with medical-related questions."

# -------------------------
# IMAGE + TEXT
# -------------------------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    session_id = request.remote_addr

    image = request.files.get("image")
    question = request.form.get("question", "").strip()

    image_description = describe_image(image) if image else ""
    combined = f"{question}\nImage observation: {image_description}".strip()

    intent = classify_intent(combined)

    # 🚨 Emergency (still allow image analysis)
    emergency_prefix = ""
    if intent == "EMERGENCY" or is_serious_medical(combined):
        emergency_prefix = emergency_message() + "\n\n"

    # ✅ Medical only
    if intent in {"MEDICAL_QUESTION", "MEDICAL_FOLLOWUP", "EMERGENCY"}:
        response = rag_chain.invoke(
            {"question": combined},
            config={"configurable": {"session_id": session_id}}
        )
        return emergency_prefix + response

    return "I can only help with medical-related images or questions."

# -------------------------
# AUDIO → TEXT
# -------------------------
@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio_route():
    audio_file = request.files.get("audio")
    if not audio_file:
        return {"error": "No audio file received"}, 400
    return transcribe_audio(audio_file)

# -------------------------
# LOCATION → HOSPITALS
# -------------------------
@app.route("/location", methods=["POST"])
def handle_location():
    lat = float(request.form["lat"])
    lng = float(request.form["lng"])

    hospitals = add_distances(lat, lng, get_nearby_hospitals(lat, lng))
    return format_hospitals(hospitals)

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)