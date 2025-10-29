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

if not openai_key and not groq_key:
    st.sidebar.warning("⚠️ Please enter at least one API key to run the app.")
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
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b"
    ]
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
uploaded_pdf = st.file_uploader("📘 Upload a PDF (max 10MB)", type=["pdf"])
web_url = st.text_input("🌐 Website URL")
query = st.text_area("💬 Enter Your Question")

if st.button("🚀 Run RAG Comparison"):
    embeddings = get_embedding_model(embedding_choice, openai_key)
    pdf_docs, web_docs = [], []
    pdf_available, web_available = False, False

    # PDF upload (limit 10MB)
    if uploaded_pdf is not None:
        if uploaded_pdf.size > 10 * 1024 * 1024:
            st.error("PDF file size exceeds 10MB limit.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.getbuffer())
            pdf_path = tmp_file.name
        pdf_docs = process_pdf(pdf_path)
        pdf_available = True if pdf_docs else False

    # Web URL
    if web_url:
        web_docs = process_web(web_url)
        web_available = True if web_docs else False

    # Combined docs
    all_docs = pdf_docs + web_docs
    if not pdf_available and not web_available:
        st.warning("No data available. Please upload a PDF or enter a Web URL.")
        st.stop()

    st.info(f"Indexing {len(all_docs)} document chunks...")

    # PDF only retriever
    if pdf_available:
        pdf_vectorstore = FAISS.from_documents(pdf_docs, embedding=embeddings)
        pdf_retriever = pdf_vectorstore.as_retriever(
            search_type=retrieval_mode,
            search_kwargs={"k": top_k}
        )
    # Web only retriever
    if web_available:
        web_vectorstore = FAISS.from_documents(web_docs, embedding=embeddings)
        web_retriever = web_vectorstore.as_retriever(
            search_type=retrieval_mode,
            search_kwargs={"k": top_k}
        )
    # Combined retriever
    combined_vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
    combined_retriever = combined_vectorstore.as_retriever(
        search_type=retrieval_mode,
        search_kwargs={"k": top_k}
    )

    st.info("⚙️ Running selected model(s)...")
    col_pdf, col_web, col_combined = st.columns(3)

    # PDF response
    with col_pdf:
        st.markdown("#### PDF Response")
        if pdf_available:
            if openai_key:
                try:
                    with st.spinner("OpenAI PDF..."):
                        pdf_openai_answer = query_retriever(
                            pdf_retriever,
                            query,
                            "OpenAI",
                            openai_model,
                            temperature,
                            openai_key
                        )
                except Exception as e:
                    pdf_openai_answer = f"Error: {e}"
                st.success(pdf_openai_answer)
            if groq_key and groq_model not in ["whisper-large-v3-turbo"]:
                try:
                    with st.spinner("Groq PDF..."):
                        pdf_groq_answer = query_retriever(
                            pdf_retriever,
                            query,
                            "Groq",
                            groq_model,
                            temperature,
                            groq_key
                        )
                except Exception as e:
                    if "model_decommissioned" in str(e):
                        pdf_groq_answer = (
                            f"Error: The selected Groq model '{groq_model}' has been decommissioned. "
                            "Please select a supported model. See: https://console.groq.com/docs/deprecations"
                        )
                    elif "does not support chat completions" in str(e):
                        pdf_groq_answer = "Groq model does not support chat completions."
                    else:
                        pdf_groq_answer = f"Error: {e}"
                import re
                cleaned_groq_answer = re.sub(r'<think>[\s\S]*?</think>', '', str(pdf_groq_answer))
                st.info(cleaned_groq_answer)
        else:
            st.info("No PDF data available.")

    # Web response
    with col_web:
        st.markdown("#### Web URL Response")
        if web_available:
            if openai_key:
                try:
                    with st.spinner("OpenAI Web..."):
                        web_openai_answer = query_retriever(
                            web_retriever,
                            query,
                            "OpenAI",
                            openai_model,
                            temperature,
                            openai_key
                        )
                except Exception as e:
                    web_openai_answer = f"Error: {e}"
                st.success(web_openai_answer)
            if groq_key and groq_model not in ["whisper-large-v3-turbo"]:
                try:
                    with st.spinner("Groq Web..."):
                        web_groq_answer = query_retriever(
                            web_retriever,
                            query,
                            "Groq",
                            groq_model,
                            temperature,
                            groq_key
                        )
                except Exception as e:
                    if "model_decommissioned" in str(e):
                        web_groq_answer = (
                            f"Error: The selected Groq model '{groq_model}' has been decommissioned. "
                            "Please select a supported model. See: https://console.groq.com/docs/deprecations"
                        )
                    elif "does not support chat completions" in str(e):
                        web_groq_answer = "Groq model does not support chat completions."
                    else:
                        web_groq_answer = f"Error: {e}"
                import re
                cleaned_groq_answer = re.sub(r'<think>[\s\S]*?</think>', '', str(web_groq_answer))
                st.info(cleaned_groq_answer)
        else:
            st.info("No Web URL data available.")

    # Combined response
    with col_combined:
        st.markdown("#### Combined PDF + Web Response")
        if pdf_available and web_available:
            if openai_key:
                try:
                    with st.spinner("OpenAI Combined..."):
                        combined_openai_answer = query_retriever(
                            combined_retriever,
                            query,
                            "OpenAI",
                            openai_model,
                            temperature,
                            openai_key
                        )
                except Exception as e:
                    combined_openai_answer = f"Error: {e}"
                st.success(combined_openai_answer)
            if groq_key and groq_model not in ["whisper-large-v3-turbo"]:
                try:
                    with st.spinner("Groq Combined..."):
                        combined_groq_answer = query_retriever(
                            combined_retriever,
                            query,
                            "Groq",
                            groq_model,
                            temperature,
                            groq_key
                        )
                except Exception as e:
                    if "model_decommissioned" in str(e):
                        combined_groq_answer = (
                            f"Error: The selected Groq model '{groq_model}' has been decommissioned. "
                            "Please select a supported model. See: https://console.groq.com/docs/deprecations"
                        )
                    elif "does not support chat completions" in str(e):
                        combined_groq_answer = "Groq model does not support chat completions."
                    else:
                        combined_groq_answer = f"Error: {e}"
                import re
                cleaned_groq_answer = re.sub(r'<think>[\s\S]*?</think>', '', str(combined_groq_answer))
                st.info(cleaned_groq_answer)
        else:
            st.info("No combined data available (need both PDF and Web URL).")

    st.markdown("---")
    st.markdown("### 🧩 Summary Insights")
    summary = []
    if openai_key:
        summary.append("- **OpenAI**: fluent and factual at low temperature.")
    if groq_key:
        summary.append("- **Groq**: faster, slightly more verbose.")
    summary.append("- Tune **temperature** or **retrieval mode** for different outputs.")
    st.write("\n".join(summary))
