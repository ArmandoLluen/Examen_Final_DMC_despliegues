import os
import re
import streamlit as st
from openai import OpenAI, OpenAIError

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # type: ignore

# 🔐 Cargar claves
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Las claves necesarias no están definidas.")
    st.stop()

# 🧠 Inicializar cliente
client = OpenAI(api_key=OPENAI_API_KEY)

# 🎨 Configurar layout
st.set_page_config(page_title="Chat con documentos", layout="centered", page_icon="📄")
st.title("📄 Asistente basado en tus documentos")

# 🗂 Inicializar estado
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = False

# ⚙️ Cachear recursos
@st.cache_resource
def init_embeddings_and_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "indice-rag"

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)
    return embeddings, index

embeddings_model, index = init_embeddings_and_index()

# 📤 Subida de archivo
uploaded_file = st.file_uploader("Sube un documento PDF", type=["pdf"], disabled=st.session_state.document_uploaded)

if uploaded_file and not st.session_state.document_uploaded:
    os.makedirs("pdf", exist_ok=True)
    file_path = os.path.join("pdf", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.document_uploaded = True
    st.success("Documento guardado correctamente.")

# 📚 Botón para indexar documentos
if st.button("Indexar documentos", disabled=st.session_state.documents_indexed):
    with st.spinner("Procesando documentos..."):
        os.makedirs("pdf", exist_ok=True)
        loader = DirectoryLoader("pdf/", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        for doc in documents:
            doc.page_content = " ".join(doc.page_content.split())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        records = []
        for i, doc in enumerate(chunks):
            source = doc.metadata.get("source", f"doc_{i}")
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', source)

            vector = embeddings_model.embed_query(doc.page_content)
            if len(vector) != 384:
                st.error(f"Vector con dimensión incorrecta: {len(vector)}")
                st.stop()

            records.append({
                "id": f"{safe_id}_chunk_{i}",
                "values": vector,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": source
                }
            })

        try:
            batch_size = 50
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                index.upsert(vectors=batch)
        except Exception as e:
            st.error(f"Error al insertar vectores: {e}")
            st.stop()

        st.session_state.retriever = Pinecone(
            index=index,
            embedding=embeddings_model,
            text_key="text"
        ).as_retriever(search_kwargs={"k": 3})

        st.session_state.documents_indexed = True
        st.success("Documentos indexados correctamente.")

# 🧠 Función de respuesta
def responder(pregunta):
    saludos = ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey", "holi", "holis", "saludos", "hello", "hi"]
    if pregunta.lower().strip() in saludos:
        return "¡Hola! ¿Qué deseas consultar sobre tus documentos?"

    if st.session_state.retriever is None:
        return "Primero debes subir e indexar tus documentos."

    docs = st.session_state.retriever.invoke(pregunta)

    if not docs:
        prompt = f"Responde de forma clara y profesional a la siguiente pregunta:\n\n{pregunta}"
    else:
        contexto = "\n\n".join([
            f"(Referencia: página {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
            for doc in docs
        ])
        prompt = f"""Responde con precisión usando el siguiente contexto. Si no es suficiente, responde con conocimiento general.
            Contexto:
            {contexto}

            Pregunta:
            {pregunta}
            """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"

# 💬 Mostrar historial
if st.session_state.chat_history:
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

# 📝 Entrada del usuario
if st.session_state.retriever:
    user_input = st.chat_input("Escribe tu pregunta...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Generando respuesta..."):
            respuesta = responder(user_input)

        with st.chat_message("assistant"):
            st.write(respuesta)

        st.session_state.chat_history.append((user_input, respuesta))
else:
    st.info("Sube e indexa tus documentos para comenzar.")
