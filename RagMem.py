# RAG + Web Search integrated app
# File: RagMem_with_search.py
import os
import uuid
import tempfile
import requests
import streamlit as st
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-family: inherit;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
from dotenv import load_dotenv
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# load .env
load_dotenv()
PG_CONN = os.getenv("PGVECTOR_DB_URL")  # e.g. postgresql://user:pass@host:5432/postgres

# helper
def uid() -> str:
    return str(uuid.uuid4())

def make_documents_from_search(results: List[Dict[str, str]], engine: str, session_id: str) -> List[Document]:
    docs: List[Document] = []
    for i, r in enumerate(results):
        text = f"{r.get('title','')}\n\n{r.get('snippet','')}\n\nURL: {r.get('link','')}"
        meta = {
            "source": r.get("link"),
            "title": r.get("title"),
            "rank": i + 1,
            "engine": engine,
            "session_id": session_id,
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def get_embedding_model(choice: str, openai_api_key: Optional[str] = None):
    choice_l = choice.lower()
    if "text-embedding-3-small" in choice_l:
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    if "text-embedding-3-large" in choice_l:
        return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Bing (Azure) and Google CSE functions
def bing_search_azure(query: str, azure_key: str, count: int = 5) -> List[Dict[str, str]]:
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": azure_key}
    params = {"q": query, "count": count, "textFormat": "Raw"}
    r = requests.get(endpoint, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({"title": item.get("name"), "snippet": item.get("snippet"), "link": item.get("url")})
    return results

def google_cse_search(query: str, google_key: str, cx: str, num: int = 5) -> List[Dict[str, str]]:
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": google_key, "cx": cx, "num": num}
    r = requests.get(endpoint, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("items", []):
        results.append({"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")})
    return results

# Streamlit UI + Session state init
st.set_page_config(page_title="RAG + Web Search (Main / Bing / Google)", layout="wide")
st.title("🧠 RagMem — Main + Bing + Google (pgvector)")

# init session state
if "main_chat" not in st.session_state:
    st.session_state.main_chat = []
if "bing_history" not in st.session_state:
    st.session_state.bing_history = []
if "google_history" not in st.session_state:
    st.session_state.google_history = []
if "bing_cache" not in st.session_state:
    st.session_state.bing_cache = {}
if "google_cache" not in st.session_state:
    st.session_state.google_cache = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "main"

tabs = st.tabs(["Main", "Bing Search", "Google Search"])

# ---- MAIN tab (keeps your existing behavior) ----
with tabs[0]:
    st.session_state.active_tab = "main"
    st.header("Main (PDF + Web RAG + Memory)")
    left, right = st.columns([2, 1])
    with left:
        uploaded_pdf = st.file_uploader("Upload PDF (Main)", type=["pdf"], key="main_pdf")
        web_url = st.text_input("Optional Web URL (Main)", key="main_web")
        user_q = st.text_area("Ask a question in Main RAG:", key="main_query")
        if st.button("Run Main RAG", key="run_main"):
            docs = []
            if uploaded_pdf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    pdf_path = tmp.name
                loader = PyPDFLoader(pdf_path)
                loaded = loader.load()
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs.extend(splitter.split_documents(loaded))
            if web_url:
                loader = UnstructuredURLLoader(urls=[web_url])
                loaded = loader.load()
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs.extend(splitter.split_documents(loaded))
            if not docs:
                st.warning("Provide PDF or Web URL for Main RAG.")
            else:
                # use OpenAI small embeddings by default; use env key if available
                emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
                try:
                    if PG_CONN:
                        vs = PGVector.from_documents(documents=docs, embedding=emb, connection_string=PG_CONN, collection_name="main_docs")
                        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                    else:
                        retriever = None
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")
                    retriever = None
                # call LLM with retriever
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
                if retriever:
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                    try:
                        ans = qa.run(user_q)
                    except Exception as e:
                        ans = f"Error during QA: {e}"
                else:
                    ans = "No retriever available (check PGVECTOR_DB_URL)."
                st.write("Answer:")
                st.write(ans)
                st.session_state.main_chat.append({"role": "user", "content": user_q})
                st.session_state.main_chat.append({"role": "assistant", "content": str(ans)})
    with right:
        st.markdown("### Main Chat History (last 10)")
        for m in st.session_state.main_chat[-10:]:
            st.write(f"**{m['role']}**: {m['content']}")

# ---- BING tab ----
with tabs[1]:
    st.session_state.active_tab = "bing"
    st.header("Bing Search (Azure)")

    bing_query = st.text_input("Bing: search query", key="bing_q")
    bing_num = st.slider("Results to fetch", 1, 10, 3, key="bing_num")
    bing_session = st.session_state.get("bing_session", uid())
    st.session_state.bing_session = bing_session

    if st.button("Run Bing Search", key="run_bing"):
        azure_key = st.sidebar.text_input("Azure Bing Key (for Bing tab)", type="password", key="azure_key_sidebar")
        if not azure_key and not os.getenv("AZURE_BING_KEY"):
            st.error("Provide Azure Bing key in sidebar or AZURE_BING_KEY in .env.")
        else:
            key_to_use = azure_key if azure_key else os.getenv("AZURE_BING_KEY")
            qk = bing_query.strip().lower()
            if qk in st.session_state.bing_cache:
                results = st.session_state.bing_cache[qk]
                st.success("Loaded results from cache.")
            else:
                try:
                    results = bing_search_azure(bing_query, key_to_use, count=bing_num)
                    st.session_state.bing_cache[qk] = results
                except Exception as e:
                    st.error(f"Bing search failed: {e}")
                    results = []
            if not results:
                st.warning("No results.")
            else:
                for r in results:
                    st.markdown(f"**{r['title']}**")
                    st.write(r["snippet"])
                    st.write(r["link"])
                docs = make_documents_from_search(results, engine="bing", session_id=bing_session)
                st.session_state["bing_last_docs"] = docs

                # persist into pgvector collection
                emb_choice = st.sidebar.selectbox("Embedding (Bing)", ["OpenAI (text-embedding-3-small)", "HuggingFace (all-MiniLM-L6-v2)"], key="bing_emb")
                emb = get_embedding_model(emb_choice, os.getenv("OPENAI_API_KEY"))
                try:
                    coll = f"search_bing_{bing_session.replace('-', '_')}"
                    pg_vs = PGVector.from_documents(documents=docs, embedding=emb, connection_string=PG_CONN, collection_name=coll)
                    retriever = pg_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                    st.success("Indexed into pgvector.")
                except Exception as e:
                    st.error(f"PGVector index error: {e}")
                    retriever = None

                # follow-up question and answer
                bing_followup = st.text_area("Ask a follow-up using Bing results as context", key="bing_followup")
                if st.button("Answer with Bing context", key="bing_answer"):
                    if not bing_followup:
                        st.warning("Enter a follow-up.")
                    else:
                        provider = st.sidebar.selectbox("LLM provider (Bing answer)", ["OpenAI", "Groq"], key="bing_provider")
                        model_choice = st.sidebar.selectbox("Model (Bing)", ["gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"], key="bing_model")
                        if provider == "OpenAI":
                            llm = ChatOpenAI(model_name=model_choice, openai_api_key=os.getenv("OPENAI_API_KEY"))
                        else:
                            llm = ChatGroq(model_name=model_choice, groq_api_key=os.getenv("GROQ_API_KEY"))
                        if retriever:
                            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                            try:
                                answer = qa.run(bing_followup)
                            except Exception as e:
                                answer = f"RetrievalQA error: {e}"
                        else:
                            top_text = "\n\n".join([d.page_content for d in docs[:3]])
                            prompt = f"Use these snippets to answer:\n\n{top_text}\n\nQuestion: {bing_followup}"
                            try:
                                answer = llm(prompt)
                            except Exception as e:
                                answer = f"LLM call error: {e}"
                        st.markdown("#### Bing-based Answer")
                        st.write(answer)
                        st.session_state.bing_history.append({"role": "user", "content": bing_followup})
                        st.session_state.bing_history.append({"role": "assistant", "content": str(answer)})
                        if st.button("Copy answer to Main Chat (Bing)", key="copy_bing"):
                            st.session_state.main_chat.append({"role": "user", "content": bing_followup})
                            st.session_state.main_chat.append({"role": "assistant", "content": str(answer)})
                            st.success("Copied to Main Chat.")

    st.markdown("### Bing tab recent history")
    for m in st.session_state.bing_history[-10:]:
        st.write(f"**{m['role']}**: {m['content']}")

# ---- GOOGLE tab ----
with tabs[2]:
    st.session_state.active_tab = "google"
    st.header("Google Search (Custom Search)")

    google_query = st.text_input("Google: search query", key="google_q")
    google_num = st.slider("Results to fetch", 1, 10, 3, key="google_num")
    google_session = st.session_state.get("google_session", uid())
    st.session_state.google_session = google_session

    if st.button("Run Google Search", key="run_google"):
        google_key = st.sidebar.text_input("Google API Key (CSE)", type="password", key="g_key_sidebar")
        google_cx = st.sidebar.text_input("Google CSE ID (cx)", key="g_cx_sidebar")
        google_key_to_use = google_key if google_key else os.getenv("GOOGLE_API_KEY")
        google_cx_to_use = google_cx if google_cx else os.getenv("GOOGLE_CSE_ID")
        if not (google_key_to_use and google_cx_to_use):
            st.error("Provide Google API key and CSE ID in sidebar or as env vars.")
        else:
            qk = google_query.strip().lower()
            if qk in st.session_state.google_cache:
                results = st.session_state.google_cache[qk]
                st.success("Loaded results from cache.")
            else:
                try:
                    results = google_cse_search(google_query, google_key_to_use, google_cx_to_use, num=google_num)
                    st.session_state.google_cache[qk] = results
                except Exception as e:
                    st.error(f"Google search failed: {e}")
                    results = []

            if results:
                for r in results:
                    st.markdown(f"**{r['title']}**")
                    st.write(r["snippet"])
                    st.write(r["link"])
                docs = make_documents_from_search(results, engine="google", session_id=google_session)
                st.session_state["google_last_docs"] = docs

                emb_choice = st.sidebar.selectbox("Embedding (Google)", ["OpenAI (text-embedding-3-small)", "HuggingFace (all-MiniLM-L6-v2)"], key="google_emb")
                emb = get_embedding_model(emb_choice, os.getenv("OPENAI_API_KEY"))
                try:
                    coll = f"search_google_{google_session.replace('-', '_')}"
                    pg_vs = PGVector.from_documents(documents=docs, embedding=emb, connection_string=PG_CONN, collection_name=coll)
                    retriever = pg_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                    st.success("Indexed into pgvector.")
                except Exception as e:
                    st.error(f"PGVector indexing error: {e}")
                    retriever = None

                google_followup = st.text_area("Ask a follow-up using Google results", key="google_followup")
                if st.button("Answer with Google context", key="google_answer"):
                    if not google_followup:
                        st.warning("Enter question.")
                    else:
                        provider = st.sidebar.selectbox("LLM provider (Google answer)", ["OpenAI", "Groq"], key="google_provider")
                        model_choice = st.sidebar.selectbox("Model (Google)", ["gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"], key="google_model")
                        if provider == "OpenAI":
                            llm = ChatOpenAI(model_name=model_choice, openai_api_key=os.getenv("OPENAI_API_KEY"))
                        else:
                            llm = ChatGroq(model_name=model_choice, groq_api_key=os.getenv("GROQ_API_KEY"))
                        if retriever:
                            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                            try:
                                answer = qa.run(google_followup)
                            except Exception as e:
                                answer = f"RetrievalQA error: {e}"
                        else:
                            top_text = "\n\n".join([d.page_content for d in docs[:3]])
                            prompt = f"Use these snippets to answer:\n\n{top_text}\n\nQuestion: {google_followup}"
                            try:
                                answer = llm(prompt)
                            except Exception as e:
                                answer = f"LLM error: {e}"
                        st.markdown("#### Google-based Answer")
                        st.write(answer)
                        st.session_state.google_history.append({"role": "user", "content": google_followup})
                        st.session_state.google_history.append({"role": "assistant", "content": str(answer)})
                        if st.button("Copy answer to Main Chat (Google)", key="copy_google"):
                            st.session_state.main_chat.append({"role": "user", "content": google_followup})
                            st.session_state.main_chat.append({"role": "assistant", "content": str(answer)})
                            st.success("Copied to Main Chat.")

    st.markdown("### Google tab recent history")
    for m in st.session_state.google_history[-10:]:
        st.write(f"**{m['role']}**: {m['content']}")

# ---------- Sidebar (dynamic - depends on active tab) ----------
st.sidebar.markdown("## App Keys & Settings")

# LLM keys (always available)
openai_key = st.sidebar.text_input("OpenAI API Key (LLM & Embeddings)", type="password", key="sidebar_openai")
groq_key = st.sidebar.text_input("Groq API Key (LLM)", type="password", key="sidebar_groq")
# fallback to env if not provided
if not openai_key:
    openai_key = os.getenv("OPENAI_API_KEY")
if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

st.sidebar.markdown(f"**PGVector:** {'configured' if PG_CONN else 'not configured'}")
active = st.session_state.get("active_tab", "main")
st.sidebar.markdown(f"**Active tab:** {active}")

# Tab-specific keys & tips
if active == "bing":
    st.sidebar.markdown("### Bing (Azure) Settings")
    azure_key = st.sidebar.text_input("Azure Bing Key", type="password", key="sidebar_azure")
    if not azure_key and os.getenv("AZURE_BING_KEY"):
        st.sidebar.info("Using AZURE_BING_KEY from env/secrets.")
elif active == "google":
    st.sidebar.markdown("### Google CSE Settings")
    gk = st.sidebar.text_input("Google API Key (CSE)", type="password", key="sidebar_google")
    gcx = st.sidebar.text_input("Google CSE ID (cx)", key="sidebar_google_cx")
    if (not gk or not gcx) and (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
        st.sidebar.info("Using GOOGLE_API_KEY / GOOGLE_CSE_ID from env/secrets.")

st.sidebar.markdown("---")
st.sidebar.markdown("Tips: Use small result counts (3) for class demos to keep latency low.")
