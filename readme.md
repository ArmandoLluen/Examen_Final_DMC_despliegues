# 💬 Chatbot RAG con Streamlit, Pinecone y OpenAI

Este proyecto implementa un chatbot conversacional basado en RAG (Retrieval-Augmented Generation), capaz de responder preguntas utilizando documentos PDF cargados por el usuario. Utiliza Streamlit como interfaz, Pinecone como base vectorial, Hugging Face para embeddings y OpenAI para generación de respuestas.

---

## 🚀 Características

- Carga automática de documentos PDF desde la carpeta `pdf/`
- División de textos en fragmentos semánticos
- Generación de embeddings con Hugging Face (`all-MiniLM-L6-v2`)
- Almacenamiento de vectores en Pinecone
- Recuperación semántica de contexto relevante
- Generación de respuestas con OpenAI (`gpt-3.5-turbo`)
- Interfaz conversacional con `st.chat_input` y `st.chat_message`
- Referencias al documento fuente en las respuestas
- Manejo de saludos y preguntas generales
- Persistencia del historial de conversación con `st.session_state`
- Optimización de recursos con `@st.cache_resource`

---

## 🛠️ Requisitos

- Python 3.10+
- Cuenta en [OpenAI](https://platform.openai.com/)
- Cuenta en [Pinecone](https://www.pinecone.io/)
- Token de Hugging Face (opcional si usas modelos públicos)

---

## 📦 Instalación

```bash
pip install -r requirements.txt
```

---

## 🧠 Créditos

Desarrollado con:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://platform.openai.com/)
- [Hugging Face](https://huggingface.co/)

---

## 👤 Autor

**Armando Lluen Gallardo**

---

## 🎓 Profesor

**Tony Trujillo**

---

## 📬 Contacto

Para dudas o mejoras, puedes abrir un issue o contribuir al repositorio.
