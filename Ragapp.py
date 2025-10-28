
import os
import streamlit as st
from dotenv import load_dotenv

# === Updated LangChain Imports ===
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq


# ==========================================
# ENV SETUP
# ==========================================
load_dotenv()
PG_CONN = os.getenv("PGVECTOR_DB_URL")

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(
    page_title="🧠 RAG Comparison: OpenAI vs Groq", layout="wide"
)
st.title("🤖 Merged RAG Comparison App (PDF + Web + pgvector)")

st.sidebar.title("⚙️ App Controls")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### 🔐 API Key Settings")
openai_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
groq_key = st.sidebar.text_input(
    "🔑 Enter your Groq API Key",
    type="password",
    placeholder="gsk-...",
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
        "HuggingFace (MiniLM-L6-v2)",
    ],
)

retrieval_mode = st.sidebar.selectbox(
    "📥 Retrieval Mode",
    ["similarity", "mmr"],
)

temperature = st.sidebar.slider(
    "🔥 Temperature (creativity)",
    0.0,
    1.0,
    0.3,
    0.1,
)

top_k = st.sidebar.slider(
    "🔍 Top K Documents",
    1,
    10,
    3,
)

openai_model = st.sidebar.selectbox(
    "🤖 OpenAI Model",
    ["gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"],
)

groq_model = st.sidebar.selectbox(
    "🦙 Groq Model",
    ["mixtral-8x7b", "llama3-70b-8192", "gemma2-9b-it"],
)

# ==========================================
# EMBEDDING MODEL SELECTOR
# ==========================================


def get_embedding_model(choice: str, openai_api_key: str):
    """Select and initialize embedding model."""
    if "small" in choice:
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=openai_api_key
        )
    if "large" in choice:
        return OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=openai_api_key
        )
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ==========================================
# DOCUMENT PROCESSING FUNCTIONS
# ==========================================


def process_pdf(pdf_path: str, embeddings):
    """Load and index a PDF into pgvector."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    vectorstore = PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="pdf_docs",
        connection_string=PG_CONN,
    )
    return vectorstore.as_retriever(
        search_type=retrieval_mode, search_kwargs={"k": top_k}
    )


def process_web(url: str, embeddings):
    """Load and index a web page into pgvector."""
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    vectorstore = PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="web_docs",
        connection_string=PG_CONN,
    )
    return vectorstore.as_retriever(
        search_type=retrieval_mode, search_kwargs={"k": top_k}
    )


# ==========================================
# RETRIEVAL + QUERY FUNCTION
# ==========================================


def query_retriever(retriever, query: str, provider: str, model_name: str, temperature: float, api_key: str):
    """Execute RAG query using selected LLM."""
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

pdf_path = st.text_input("📘 PDF File Path")
web_url = st.text_input("🌐 Website URL")
query = st.text_area("💬 Enter Your Question")

if st.button("🚀 Run RAG Comparison"):
    if not PG_CONN:
        st.error("❌ PGVECTOR_DB_URL not found. Add it to your .env file.")
        st.stop()

    embeddings = get_embedding_model(embedding_choice, openai_key)
    retriever = None

    # Process PDF
    if pdf_path and os.path.isfile(pdf_path):
        pdf_retriever = process_pdf(pdf_path, embeddings)
        retriever = pdf_retriever

    # Process Web URL
    if web_url:
        web_retriever = process_web(web_url, embeddings)
        if retriever:
            retriever = (
                retriever.vectorstore.merge_from([web_retriever.vectorstore])
                .as_retriever(
                    search_type=retrieval_mode, search_kwargs={"k": top_k}
                )
            )
        else:
            retriever = web_retriever

    if not retriever:
        st.warning("Please provide either a PDF or a Web URL.")
        st.stop()

    st.info("⚙️ Running both models in parallel...")

    # Run OpenAI and Groq models
    with st.spinner("Running OpenAI Model..."):
        openai_answer = query_retriever(
            retriever, query, "OpenAI", openai_model, temperature, openai_key
        )

    with st.spinner("Running Groq Model..."):
        groq_answer = query_retriever(
            retriever, query, "Groq", groq_model, temperature, groq_key
        )

    # ==========================================
    # SIDE-BY-SIDE DISPLAY
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🤖 OpenAI ({openai_model})")
        st.success(openai_answer)
    with col2:
        st.markdown(f"### 🦙 Groq ({groq_model})")
        st.info(groq_answer)

    # Optional summary
    st.markdown("---")
    st.markdown("### 🧩 Summary Insights")
    st.write(
        """
        - **OpenAI** tends to be more fluent and factual at lower temperatures.
        - **Groq** (especially Mixtral) is faster and slightly more verbose.
        - Try adjusting **temperature** or switching to **MMR retrieval** to
          reduce redundancy.
        """
    )
