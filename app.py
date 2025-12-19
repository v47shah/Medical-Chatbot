from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import chat_prompt
from src.prompt import *


from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

load_dotenv()

from src.classifier import classify_intent

# --------------------
# Conversation State
# --------------------

conversation_state = {}

def has_active_medical_context(session_id: str) -> bool:
    return conversation_state.get(session_id, False)

def set_medical_context(session_id: str):
    conversation_state[session_id] = True


# --------------------
# Chat History Store
# --------------------
from langchain_core.chat_history import InMemoryChatMessageHistory

chat_histories = {}

def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# --------------------
# App + ENV
# --------------------
app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --------------------
# Vector Store
# --------------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


# --------------------
# LLM + Prompt
# --------------------
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)


# --------------------
# LCEL RAG Chain 
# --------------------
from langchain_core.runnables import RunnablePassthrough

base_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["question"])
    )
    | chat_prompt
    | chat_model
    | StrOutputParser()
)



rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    return render_template("chat.html")


# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form["msg"]

#     if not is_medical_question(msg):
#         return "I can only answer medical-related questions."
    
#     session_id = request.form.get("session_id", "default")

#     print("User:", msg)

#     answer = rag_chain.invoke(
#         {"question": msg},
#         config={"configurable": {"session_id": session_id}}
#     )

#     print("Response:", answer)
#     return str(answer)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    session_id = request.remote_addr  # simple session key

    intent = classify_intent(msg)

    # 1️⃣ Greeting
    if intent == "GREETING":
        return (
            "Hi! 👋 I’m a medical chatbot. "
            "I can help answer questions about symptoms, diseases, "
            "treatments, and first aid."
        )

    # 2️⃣ Medical question
    if intent == "MEDICAL_QUESTION":
        set_medical_context(session_id)

        docs = retriever.invoke(msg)
        if not docs:
            return "I don't know based on the provided medical documents."

        return rag_chain.invoke(
            {"question": msg},
            config={"configurable": {"session_id": session_id}}
        )   

    # 3️⃣ Medical follow-up (conversational!)
    if intent == "MEDICAL_FOLLOWUP":
        if not has_active_medical_context(session_id):
            return (
                "Could you please provide more details about the medical "
                "issue you're referring to?"
            )

        docs = retriever.invoke(msg)
        if not docs:
            return "I don't know based on the provided medical documents."

        return rag_chain.invoke(
            {"question": msg},
            config={"configurable": {"session_id": session_id}}
        )   

    # 4️⃣ Out of scope
    return (
        "I can only help with medical-related questions. "
        "Please ask about symptoms, treatments, or first aid."
    )


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
