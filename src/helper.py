from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.prompt import IMAGE_DESCRIPTION_PROMPT


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

#############################################################################
# Image Description Function
#############################################################################
import base64
from openai import OpenAI
import os

vision_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def describe_image(image_file):
    """
    Returns a neutral, visible-only description of the image.
    """
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = vision_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

#############################################################################
# Audio Transcription Function (Whisper – English only)
#############################################################################
from openai import OpenAI
import tempfile
import os

audio_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_file):
    """
    Transcribes English audio using Whisper.
    Returns:
        {
            "text": "..."
        }
    """
    # Save temp file (browser mic usually sends webm)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    try:
        with open(temp_path, "rb") as f:
            transcription = audio_client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                language="en"   # 🔒 ENGLISH ONLY
            )

        return {"text": transcription.text}

    finally:
        os.remove(temp_path)

