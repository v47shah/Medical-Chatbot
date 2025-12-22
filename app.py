from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import requests

from src.helper import download_hugging_face_embeddings
from src.prompt import chat_prompt
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



# # =========================================================
# # App + ENV
# # =========================================================

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_MAPS_API_KEY"] = GOOGLE_MAPS_API_KEY

# # =========================================================
# # Conversation State (per session)
# # =========================================================
conversation_state = {}

def reset_state(session_id):
    conversation_state[session_id] = {
        "has_medical_context": False,
        "awaiting_location": False
    }

def get_state(session_id):
    if session_id not in conversation_state:
        reset_state(session_id)
    return conversation_state[session_id]

def set_awaiting_location(session_id, value=True):
    get_state(session_id)["awaiting_location"] = value

def is_awaiting_location(session_id):
    return get_state(session_id)["awaiting_location"]

def set_medical_context(session_id):
    get_state(session_id)["has_medical_context"] = True

def has_medical_context(session_id):
    return get_state(session_id)["has_medical_context"]

# # =========================================================
# # Google Maps Helpers
# # =========================================================
def geocode_location(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_MAPS_API_KEY}
    res = requests.get(url, params=params).json()

    if not res.get("results"):
        return None

    loc = res["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

def get_nearby_hospitals(lat, lng):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 5000,  # tighter radius = better relevance
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
        element = data["rows"][0]["elements"][i]
        if element["status"] != "OK":
            continue

        h["distance"] = element["distance"]["text"]
        h["duration"] = element["duration"]["text"]
        h["duration_value"] = element["duration"]["value"]  # seconds
        enriched.append(h)

    # 🔥 SORT BY TRAVEL TIME
    enriched.sort(key=lambda x: x["duration_value"])

    # 🔥 TAKE CLOSEST 3
    return enriched[:5]


def format_hospitals(hospitals):
    if not hospitals:
        return "I couldn’t find nearby hospitals for that location."

    lines = ["🏥 Nearest hospitals to you:\n"]
    for i, h in enumerate(hospitals, 1):
        lines.append(
            f"{i}️⃣ {h.get('name')}\n"
            f"📍 {h.get('vicinity')}\n"
            f"📏 {h.get('distance')} • ⏱ {h.get('duration')}\n"
        )
    return "\n".join(lines)

# # =========================================================
# # Emergency Detection (rule-based, safe)
# # =========================================================
def is_serious_medical(msg):
    msg = msg.lower()
    red_flags = [
        "chest pain", "shortness of breath", "can't breathe",
        "difficulty breathing", "stroke", "seizure",
        "unconscious", "passed out", "fainted",
        "severe bleeding", "bleeding heavily",
        "overdose", "poisoning", "anaphylaxis",
        "throat closing", "suicidal", "kill myself"
    ]
    return any(flag in msg for flag in red_flags)

# # =========================================================
# # Vector Store + RAG
# # =========================================================
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
# Routes
# =========================================================
@app.route("/")
def index():
    session_id = request.remote_addr
    reset_state(session_id)  # 🔥 RESET MEMORY ON REFRESH
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    session_id = request.remote_addr

    # ✅ 1. ADDRESS MODE — skip classifier entirely
    if is_awaiting_location(session_id):
        coords = geocode_location(msg)
        if not coords:
            return "I couldn’t understand that address. Please try again."

        lat, lng = coords
        hospitals = get_nearby_hospitals(lat, lng)
        hospitals = add_distances(lat, lng, hospitals)

        set_awaiting_location(session_id, False)
        return format_hospitals(hospitals)

    # ✅ 2. Normal intent classification
    intent = classify_intent(msg)

    # 🚨 Emergency
    if intent == "EMERGENCY" or is_serious_medical(msg):
        set_medical_context(session_id)
        set_awaiting_location(session_id, True)
        return (
            "⚠️ This may be a medical emergency.\n\n"
            "If this is urgent, please call your local emergency number immediately.\n\n"
            "If you want, type your address or city and I’ll find the nearest hospitals."
        )

    # 👋 Greeting
    if intent == "GREETING":
        return (
            "Hi! 👋 I’m a medical chatbot.\n"
            "I can help with symptoms, conditions, treatments, and emergencies."
        )

    # 🩺 Medical question
    if intent == "MEDICAL_QUESTION":
        set_medical_context(session_id)
        return rag_chain.invoke(
            {"question": msg},
            config={"configurable": {"session_id": session_id}}
        )

    # 🔁 Follow-up
    if intent == "MEDICAL_FOLLOWUP":
        if not has_medical_context(session_id):
            return "Could you provide more details about the medical issue?"
        return rag_chain.invoke(
            {"question": msg},
            config={"configurable": {"session_id": session_id}}
        )

    # 🚫 Out of scope
    return (
        "I can only help with medical-related questions.\n"
        "Please ask about symptoms, treatments, or emergencies."
    )

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

