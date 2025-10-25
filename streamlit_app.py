import os
import streamlit as st
from openai import OpenAI, OpenAIError

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec # type: ignore

# 🌱 Cargar variables de entorno
# 🔐 Acceder a las claves desde secrets.toml
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 🔐 Validar claves API
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Las claves API no están definidas en el archivo .env")
    st.stop()

# 🧠 Inicializar cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 🎨 Configurar layout principal
st.set_page_config(page_title="Chatbot RAG", layout="centered", page_icon="💬")
st.title("💬 Chatbot experto en tus documentos")

# 🗂 Inicializar historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ⚙️ Cachear recursos
@st.cache_resource
def load_retriever():
    # 📁 Verificar carpeta de PDFs
    if not os.path.exists("pdf"):
        os.makedirs("pdf")
        st.warning("📂 Carpeta 'pdf/' creada. Agrega documentos PDF para comenzar.")

    # 📄 Cargar documentos
    loader = DirectoryLoader("pdf/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        st.warning("⚠️ No se encontraron documentos PDF en la carpeta 'pdf/'.")
        return None

    # 🧹 Limpiar texto
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())

    # ✂️ Dividir en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs_chunks = text_splitter.split_documents(documents)

    # 🔗 Inicializar Pinecone
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "indice-huggingface"

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    # 🧠 Embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 📦 Insertar vectores en Pinecone
    records = []
    for i, doc in enumerate(docs_chunks):
        emb_vector = embeddings_model.embed_query(doc.page_content)
        record = {
            "id": f"chunk_{i}",
            "values": emb_vector,
            "metadata": {
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "")
            }
        }
        records.append(record)

    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert(vectors=batch)

    # 🔍 Crear retriever
    retriever = Pinecone(
        index=index,
        embedding=embeddings_model,
        text_key="text"
    ).as_retriever(search_kwargs={"k": 3})

    return retriever

retriever = load_retriever()

# 🧠 Función RAG con manejo de saludos
def ask_rag_openai(question):
    # 🗣️ Detectar saludos comunes
    saludos = [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
        "qué tal", "hey", "holi", "holis", "saludos", "hello", "hi"
    ]
    if question.lower().strip() in saludos:
        return "¡Hola! 👋 Soy tu asistente experto en documentos. ¿Qué quieres consultar hoy?"

    # ⚠️ Verificar si el retriever está disponible
    if retriever is None:
        return "⚠️ No hay documentos cargados para responder preguntas. Agrega PDFs a la carpeta 'pdf/'."

    # 🔍 Recuperar fragmentos relevantes desde Pinecone
    docs = retriever.invoke(question)

    # 🧠 Si no hay contexto relevante, dejar que el LLM responda con conocimiento general
    if not docs:
        prompt_general = f"""
            Eres un asistente experto. Responde de forma clara, precisa y profesional a la siguiente pregunta:

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

    # 🧩 Construir contexto con referencias al documento fuente
    context = "\n\n".join([
        f"(Referencia: {doc.metadata.get('source', 'desconocido')} - página {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
        for doc in docs
    ])

    # 📝 Prompt optimizado para respuestas fieles al contexto
    prompt = f"""
        Eres un experto en la documentación proporcionada. Responde de forma clara, precisa y profesional.
        Incluye referencias al documento fuente si es posible. Solo si el contexto no es suficiente, responde con tu conocimiento general.

        Contexto:
        {context}

        Pregunta:
        {question}
        """

    # 🤖 Generar respuesta con OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# 💬 Mostrar historial conversacional
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# 📝 Capturar nuevo mensaje
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    answer = ask_rag_openai(user_input)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append((user_input, answer))

uploaded_files = st.file_uploader(
    "📤 Sube tus documentos PDF aquí",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("pdf", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("pdf", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"✅ {len(uploaded_files)} archivo(s) guardado(s) en la carpeta 'pdf/'. Procesando en Pinecone...")

    # 🔄 Ejecutar procesamiento en Pinecone automáticamente
    retriever = load_retriever()
    st.success("📚 Documentos indexados correctamente. Ya puedes hacer preguntas.")