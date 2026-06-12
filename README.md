# 🧠 PDF RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it — no API keys needed.

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDF (text extraction)
    │
    ▼
Chunking (sliding window, 200 words, 40-word overlap)
    │
    ▼
all-MiniLM-L6-v2 (sentence embeddings)
    │
    ▼
FAISS IndexFlatIP (cosine similarity search)
    │
    ▼
Top-5 chunks → deepset/roberta-base-squad2 (QA)
    │
    ▼
Answer + source chunks displayed in Streamlit UI
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
cd rag_pdf_chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads two models (~90 MB total):
> - `sentence-transformers/all-MiniLM-L6-v2`
> - `deepset/roberta-base-squad2`
>
> They are cached locally after the first download.

---

## 🚀 Run

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📖 How to use

1. **Upload** a PDF using the sidebar uploader.
2. Click **⚡ Process Document** — text is extracted, chunked, and indexed.
3. Type a question in the chat input and hit **Send**.
4. See the answer and the retrieved source chunks side-by-side.
5. Ask follow-up questions — the engine uses recent chat history for context.
6. Click **🗑️ Clear Chat** to start a new conversation on the same document.
7. Upload a new PDF anytime to switch documents.

---

## 🧩 Key concepts

| Concept | What it does |
|---|---|
| **RAG** | Retrieves relevant chunks before answering — avoids hallucination |
| **Chunking** | Splits large text into manageable, overlapping windows |
| **Embeddings** | Converts text to vectors that capture semantic meaning |
| **FAISS** | Fast approximate nearest-neighbour search over embeddings |
| **Extractive QA** | RoBERTa extracts a precise span from the context — grounded answers |
| **Chat history** | Last assistant reply is injected into the retrieval query for follow-ups |

---

## 📁 Project structure

```
rag_pdf_chatbot/
├── app.py            # Streamlit UI
├── rag_engine.py     # RAG pipeline (PDF → chunks → embed → FAISS → QA)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 🔧 Tuning knobs (in rag_engine.py)

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 200 words | Larger = more context per chunk, slower |
| `CHUNK_OVERLAP` | 40 words | More overlap = fewer missed boundaries |
| `TOP_K` | 5 | More chunks = more context, may dilute answer |

---

## 💡 Tips

- Works best on **text-based PDFs** (research papers, manuals, reports). Scanned PDFs without OCR will give poor results.
- Ask **specific questions** rather than vague ones — e.g. "What is the proposed method in section 3?" vs "Explain the paper".
- If confidence is low (< 0.05), the model will say so rather than guess.
