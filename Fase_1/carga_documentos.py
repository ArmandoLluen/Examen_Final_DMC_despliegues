# CARGA DE DOCUMENTOS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
# DIVISIÓN DE TEXTOS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# EMBEDDINGS (Hugging Face)
from langchain_huggingface import HuggingFaceEmbeddings

# BASE VECTORIAL (Pinecone)
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# UTILIDADES
import os

# Configuración de Pinecone
# API Keys
PINECONE_API_KEY = "pcsk_oQEHh_KkpDbvQ7WYWi177E8am5aWiBDoWZpqL6fhPrbpGXqnJhjCSwuqZn9jzRc23Giot"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "indice-huggingface"

# Crear el índice si no existe
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # tamaño del embedding del modelo
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Índice creado en Pinecone.")
else:
    print("Índice ya existente.")

index = pc.Index(index_name)


#CARGA Y PREPARACIÓN DE DOCUMENTOS (LangChain)
# Carga automática de todos los PDF de la carpeta 'pdf/'
loader = DirectoryLoader("pdf/", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print(f"Documentos cargados: {len(documents)}")

# Limpiar texto
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split())

# Dividir textos en fragmentos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,    # tamaño del fragmento
    chunk_overlap=20,  # solapamiento entre fragmentos
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

docs_chunks = text_splitter.split_documents(documents)
print(f"Total de fragmentos creados: {len(docs_chunks)}")

#GENERAR EMBEDDINGS CON HUGGING FACE
# Usaremos el modelo de Hugging Face "all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generar embeddings y preparar registros para Pinecone
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

# Almacenar los embeddings en Pinecone (upsert)
batch_size = 50
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Almacenados {i + len(batch)} de {len(records)} registros en Pinecone.")

# Confirmación final
print("Todos los embeddings han sido almacenados en Pinecone.")

# CONSULTA DE PRUEBA
def text_to_vector(query_text):
    return embeddings_model.embed_query(query_text)

query = "Qué es el cyberbullying?"
query_vector = text_to_vector(query)

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\n🔎 Resultados más relevantes:")
for match in results['matches']:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Texto: {match['metadata']['text'][:300]}...")