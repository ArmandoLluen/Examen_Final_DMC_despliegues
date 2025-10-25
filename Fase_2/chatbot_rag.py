import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# 🌱 Cargar variables de entorno
load_dotenv()

# 🔐 Validar claves API
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Las claves API no están definidas en el archivo .env")
    st.stop()

# ⚙️ Cachear recursos
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Inicializar cliente Pinecone
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "indice-huggingface"

    # Crear índice si no existe
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)  # ✅ compatible con LangChain

    retriever = Pinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    ).as_retriever(search_kwargs={"k": 3})

    return retriever

retriever = load_retriever()

# 🧠 Función RAG manual con OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def ask_rag_openai(question):
    saludos = [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
        "qué tal", "hey", "holi", "holis", "saludos", "hello", "hi"
    ]
    if question.lower().strip() in saludos:
        return "¡Hola! 👋 Soy tu asistente experto en documentos. ¿Qué quieres consultar hoy?"

    docs = retriever.invoke(question)

    if not docs:
        return "No encontré información relevante en los documentos. ¿Puedes reformular tu pregunta?"

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
        Eres un experto en la documentación proporcionada. Responde de forma clara, precisa y profesional.
        Contexto:
        {context}

        Pregunta:
        {question}
        """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# 🎨 Interfaz Streamlit
st.set_page_config(page_title="Chatbot RAG", layout="centered")
st.title("📚 Chatbot experto en tus documentos")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🗂 Mostrar historial en orden
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# 💬 Capturar nuevo input
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    answer = ask_rag_openai(user_input)

    with st.chat_message("assistant"):
        st.write(answer)

    # 🧠 Guardar en historial
    st.session_state.chat_history.append((user_input, answer))
