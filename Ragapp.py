import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# === LangChain Imports ===
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

# ==========================================
# ENV SETUP
# ==========================================

load_dotenv()

# ==========================================
# STREAMLIT CONFIG
# ==========================================

st.set_page_config(
    page_title="🧠 RAG Comparison: OpenAI vs Groq",
    layout="wide"
)
st.title("🤖 Merged RAG Comparison App (PDF + Web)")


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown("### 🔐 API Key Settings")
openai_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password",
    placeholder="sk-..."
)
groq_key = st.sidebar.text_input(
    "🔑 Enter your Groq API Key",
    type="password",
    placeholder="gsk-..."
)

# Fallback for local dev if user didn't provide
if not openai_key:
    openai_key = os.getenv("OPENAI_API_KEY")
if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if not openai_key or not groq_key:
    st.sidebar.warning("⚠️ Please enter both API keys to run the comparison.")
    st.stop()

embedding_choice = st.sidebar.selectbox(
    "🧬 Embedding Model",
    [
        "OpenAI (text-embedding-3-small)",
        "OpenAI (text-embedding-3-large)",
        "HuggingFace (MiniLM-L6-v2)"
    ]
)

retrieval_mode = st.sidebar.selectbox(
    "📥 Retrieval Mode",
    ["similarity", "mmr"]
)

temperature = st.sidebar.slider(
    "🔥 Temperature (creativity)",
    0.0,
    1.0,
    0.3,
    0.1
)

top_k = st.sidebar.slider(
    "🔍 Top K Documents",
    1,
    10,
    3
)

openai_model = st.sidebar.selectbox(
    "🤖 OpenAI Model",
    ["gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"]
)
groq_model = st.sidebar.selectbox(
    "🦙 Groq Model",
    ["mixtral-8x7b", "llama3-70b-8192", "gemma2-9b-it"]
)

# ==========================================
# EMBEDDING MODEL SELECTOR
# ==========================================


def get_embedding_model(choice: str, openai_api_key: str):
    if "small" in choice:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
    if "large" in choice:
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# DOCUMENT PROCESSING FUNCTIONS (FAISS)
# ==========================================


def process_pdf(pdf_path: str):
    st.info("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def process_web(url: str):
    st.info("Loading web page...")
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ==========================================
# RETRIEVAL + QUERY FUNCTION
# ==========================================


def query_retriever(
    retriever,
    query: str,
    provider: str,
    model_name: str,
    temperature: float,
    api_key: str
):
    if provider == "OpenAI":
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
    elif provider == "Groq":
        llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=api_key
        )
    else:
        raise ValueError("Invalid provider")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(query)

# ==========================================
# MAIN APP
# ==========================================

st.subheader("📄 Upload PDF and/or Enter Web URL")
uploaded_pdf = st.file_uploader("📘 Upload a PDF", type=["pdf"])
web_url = st.text_input("🌐 Website URL")
query = st.text_area("💬 Enter Your Question")

if st.button("🚀 Run RAG Comparison"):
    embeddings = get_embedding_model(embedding_choice, openai_key)
    all_docs = []

    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_pdf.getbuffer())
            pdf_path = tmp_file.name
        all_docs.extend(process_pdf(pdf_path))

    if web_url:
        all_docs.extend(process_web(web_url))

    if not all_docs:
        st.warning("Please upload a PDF or enter a Web URL.")
        st.stop()

    st.info(f"Indexing {len(all_docs)} document chunks...")
    vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type=retrieval_mode,
        search_kwargs={"k": top_k}
    )

    st.info("⚙️ Running both models in parallel...")

    try:
        with st.spinner("Running OpenAI Model..."):
            openai_answer = query_retriever(
                retriever,
                query,
                "OpenAI",
                openai_model,
                temperature,
                openai_key
            )
    except Exception as e:
        openai_answer = f"Error: {e}"

    try:
        with st.spinner("Running Groq Model..."):
            groq_answer = query_retriever(
                retriever,
                query,
                "Groq",
                groq_model,
                temperature,
                groq_key
            )
    except Exception as e:
        groq_answer = f"Error: {e}"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🤖 OpenAI ({openai_model})")
        st.success(openai_answer)
    with col2:
        st.markdown(f"### 🦙 Groq ({groq_model})")
        st.info(groq_answer)

    st.markdown("---")
    st.markdown("### 🧩 Summary Insights")
    st.write("""
    - **OpenAI**: fluent and factual at low temperature.
    - **Groq**: faster, slightly more verbose.
    - Tune **temperature** or **retrieval mode** for different outputs.
    """)
