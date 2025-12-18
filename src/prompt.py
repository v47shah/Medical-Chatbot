from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Medical assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "keep the answer concise.\n\n"
        "{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
