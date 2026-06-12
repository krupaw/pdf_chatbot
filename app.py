import uuid
from datetime import datetime

import streamlit as st
from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (light theme) ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f7f8fa; color: #1f2330; }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] * { color: #1f2330; }

    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 16px;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
    }

    .user-msg {
        background: #2563eb;
        color: #fff;
        padding: 10px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 8px 0 8px 20%;
        font-size: 14px;
        line-height: 1.6;
    }

    .ai-msg {
        background: #f1f3f8;
        color: #1f2330;
        padding: 10px 16px;
        border-radius: 16px 16px 16px 4px;
        margin: 8px 20% 8px 0;
        font-size: 14px;
        line-height: 1.6;
        border-left: 3px solid #7c3aed;
    }

    .chunk-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #4b5563;
        line-height: 1.5;
    }

    .stat-pill {
        display: inline-block;
        background: #f1f3f8;
        border: 1px solid #e5e7eb;
        border-radius: 99px;
        padding: 4px 12px;
        font-size: 12px;
        color: #4b5563;
        margin: 4px 4px 4px 0;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        color: #1f2330 !important;
        padding: 12px 16px !important;
    }

    .stButton > button {
        background-color: #7c3aed !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover { background-color: #6d28d9 !important; }

    section[data-testid="stSidebar"] .stButton > button {
        background-color: #f1f3f8 !important;
        color: #1f2330 !important;
        text-align: left !important;
        font-weight: 400 !important;
        border: 1px solid #e5e7eb !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #e5e7eb !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1.5px dashed #d1d5db;
        border-radius: 12px;
        padding: 20px;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #f7f8fa; }
    ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    h1, h2, h3 { color: #1f2330 !important; }

    .center-upload {
        max-width: 560px;
        margin: 40px auto;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def new_session() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "created": datetime.now().strftime("%H:%M"),
        "engine": None,
        "doc_info": None,
        "chat_history": [],
        "last_result": None,
    }


if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "active_session" not in st.session_state:
    sess = new_session()
    st.session_state.sessions[sess["id"]] = sess
    st.session_state.active_session = sess["id"]


def get_active() -> dict:
    return st.session_state.sessions[st.session_state.active_session]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PDF RAG Chatbot")
    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        sess = new_session()
        st.session_state.sessions[sess["id"]] = sess
        st.session_state.active_session = sess["id"]
        st.rerun()

    if st.button("🗑️ Clear History (this chat)", use_container_width=True):
        active = get_active()
        active["chat_history"] = []
        active["last_result"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("**💬 Previous Chats**")

    for sid, sess in reversed(list(st.session_state.sessions.items())):
        label = sess["title"]
        if sess["doc_info"]:
            label = f"📄 {sess['doc_info']['name']}"
        is_active = sid == st.session_state.active_session
        prefix = "➡️ " if is_active else ""
        if st.button(f"{prefix}{label}  ·  {sess['created']}", key=f"sess_{sid}", use_container_width=True):
            st.session_state.active_session = sid
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **How it works**
    1. Upload & process PDF
    2. Text → chunks → embeddings
    3. FAISS similarity search
    4. RoBERTa QA model answers
    """)
    st.markdown("""
    <div style='font-size:11px;color:#9ca3af;margin-top:8px;'>
    all-MiniLM-L6-v2 · deepset/roberta-base-squad2 · FAISS
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
active = get_active()

if active["engine"] is None:
    st.markdown("<div class='center-upload'>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload a PDF to get started")
    st.markdown("<p style='color:#6b7280;'>Ask questions about any PDF — research papers, manuals, reports, and more.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Extracting text…"):
                engine = RAGEngine()
                info = engine.load_pdf(uploaded_file)

            with st.spinner("Building embeddings & FAISS index…"):
                engine.build_index()
                info["chunks"] = len(engine.chunks)

            active["engine"] = engine
            active["doc_info"] = info
            active["title"] = info["name"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

else:
    info = active["doc_info"]
    st.markdown(f"### 💬 {info['name']}")
    st.markdown(
        f"<span class='stat-pill'>📃 {info['pages']} pages</span> "
        f"<span class='stat-pill'>🧩 {info['chunks']} chunks</span> "
        f"<span class='stat-pill'>📝 {info['words']:,} words</span>",
        unsafe_allow_html=True,
    )

    col_main, col_chunks = st.columns([3, 2])

    with col_main:
        chat_html = "<div class='chat-container'>"
        if not active["chat_history"]:
            chat_html += "<div style='text-align:center;color:#9ca3af;font-size:13px;padding:40px 0;'>Ask a question about the document below 👇</div>"
        else:
            for turn in active["chat_history"]:
                if turn["role"] == "user":
                    chat_html += f"<div class='user-msg'>🧑 {turn['content']}</div>"
                else:
                    chat_html += f"<div class='ai-msg'>🤖 {turn['content']}</div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                query = st.text_input(
                    "Ask a question",
                    placeholder="What is this document about?",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Send")

        if submitted and query:
            with st.spinner("Thinking…"):
                result = active["engine"].answer(query, chat_history=active["chat_history"])
            active["chat_history"].append({"role": "user", "content": query})
            active["chat_history"].append({"role": "assistant", "content": result["answer"]})
            active["last_result"] = result
            st.rerun()

    with col_chunks:
        st.markdown("#### 🔍 Retrieved Chunks")
        result = active["last_result"]
        if result:
            st.markdown(
                f"<span class='stat-pill'>⚡ {result['score']:.2f} confidence</span> "
                f"<span class='stat-pill'>🧩 top {len(result['chunks'])} chunks</span>",
                unsafe_allow_html=True,
            )
            for i, chunk in enumerate(result["chunks"], 1):
                preview = chunk[:280].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class='chunk-card'>
                    <div style='color:#7c3aed;font-size:12px;font-weight:600;margin-bottom:4px;'>Chunk #{i}</div>
                    {preview}{'…' if len(chunk) > 280 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='text-align:center;color:#9ca3af;font-size:13px;padding:40px 0;'>"
                "Retrieved context chunks will appear here after you ask a question."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.button("📤 Upload a different PDF"):
        active["engine"] = None
        active["doc_info"] = None
        active["chat_history"] = []
        active["last_result"] = None
        st.rerun()
