import os
import re
import streamlit as st
from openai import OpenAI, OpenAIError

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # type: ignore

# 🔐 Cargar claves desde secrets.toml
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Las claves API no están definidas.")
    st.stop()

# 🧠 Inicializar cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 🎨 Configurar layout
st.set_page_config(page_title="Chatbot RAG", layout="centered", page_icon="💬")
st.title("💬 Chatbot experto en tus documentos")

# 🗂 Inicializar historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📦 Inicializar retriever si ya fue cargado
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ⚙️ Cachear embeddings y conexión Pinecone
@st.cache_resource
def init_embeddings_and_index():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
    return embeddings_model, index

embeddings_model, index = init_embeddings_and_index()

# 📤 Subida de archivo
uploaded_file = st.file_uploader("📤 Sube un documento PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("pdf", exist_ok=True)
    file_path = os.path.join("pdf", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Archivo guardado: {uploaded_file.name}")

# 📚 Botón para cargar todos los PDFs en Pinecone
if st.button("📚 Cargar documentos PDF en Pinecone"):
    os.makedirs("pdf", exist_ok=True)
    loader = DirectoryLoader("pdf/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs_chunks = text_splitter.split_documents(documents)

    records = []
    for i, doc in enumerate(docs_chunks):
        source_name = doc.metadata.get("source", f"doc_{i}")
        safe_id_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', source_name)

        emb_vector = embeddings_model.embed_query(doc.page_content)
        if len(emb_vector) != 384:
            st.error(f"❌ Vector con dimensión incorrecta: {len(emb_vector)}")
            st.stop()

        record = {
            "id": f"{safe_id_prefix}_chunk_{i}",
            "values": emb_vector,
            "metadata": {
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": source_name
            }
        }
        records.append(record)

    try:
        batch_size = 50
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            index.upsert(vectors=batch)
    except Exception as e:
        st.error(f"❌ Error al insertar vectores en Pinecone: {e}")
        st.stop()

    st.session_state.retriever = Pinecone(
        index=index,
        embedding=embeddings_model,
        text_key="text"
    ).as_retriever(search_kwargs={"k": 3})

    st.success("✅ Todos los documentos PDF han sido indexados correctamente.")

# 🧠 Función RAG
def ask_rag_openai(question):
    saludos = [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
        "qué tal", "hey", "holi", "holis", "saludos", "hello", "hi"
    ]
    if question.lower().strip() in saludos:
        return "¡Hola! 👋 Soy tu asistente experto en documentos. ¿Qué quieres consultar hoy?"

    if st.session_state.retriever is None:
        return "⚠️ Primero debes cargar documentos PDF en Pinecone usando el botón."

    docs = st.session_state.retriever.invoke(question)

    if not docs:
        prompt_general = f"""
        Eres un asistente experto. Responde de forma clara, precisa y profesional a la siguiente pregunta:

        {question}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_general}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            return f"❌ Error al generar la respuesta: {str(e)}"
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"

    context = "\n\n".join([
        f"(Referencia: {doc.metadata.get('source', 'desconocido')} - página {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
    Eres un experto en la documentación proporcionada. Responde de forma clara, precisa y profesional.
    Incluye referencias al documento fuente si es posible. Solo si el contexto no es suficiente, responde con tu conocimiento general.

    Contexto:
    {context}

    Pregunta:
    {question}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ Error al generar la respuesta: {str(e)}"
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

# 💬 Mostrar historial conversacional
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

        answer = ask_rag_openai(user_input)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append((user_input, answer))
else:
    st.info("📂 Por favor, sube documentos PDF y presiona el botón para indexarlos.")
