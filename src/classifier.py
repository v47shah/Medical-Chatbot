from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cheap, deterministic model
classifier_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

classifier_prompt = ChatPromptTemplate.from_template(
    """
You are an intent classifier for a medical chatbot.

Classify the user's message into ONE of the following intents:
- GREETING
- MEDICAL_QUESTION
- MEDICAL_FOLLOWUP
- EMERGENCY
- OUT_OF_SCOPE

Rules:
- Life-threatening symptoms (chest pain, difficulty breathing, stroke symptoms,
  severe bleeding, loss of consciousness, suicidal thoughts) → EMERGENCY
- Health-related questions → MEDICAL_QUESTION
- Short follow-ups → MEDICAL_FOLLOWUP
- Greetings → GREETING
- Anything else → OUT_OF_SCOPE

Respond with ONLY the intent name.

Message: {message}
"""
)


classifier_chain = (
    classifier_prompt
    | classifier_llm
    | StrOutputParser()
)

def classify_intent(message: str) -> str:
    return classifier_chain.invoke({"message": message}).strip().upper()

