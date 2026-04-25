"""
Mini NotebookLM Clone - Flask Backend
v6: Ollama (phi model) for answer generation

Stack:
  PDF parsing  → pypdf
  Chunking     → plain Python (word-count split)
  Embeddings   → TF-IDF via scikit-learn
  Vector store → FAISS (cosine search on normalised vectors)
  Answering    → Ollama local LLM (phi model) via HTTP API
"""

import io
import requests
import numpy as np
import faiss
import pypdf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────

app = Flask(__name__)
CORS(app)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi"

# ─────────────────────────────────────────
# In-memory RAG state
# ─────────────────────────────────────────

text_chunks: list[str] = []
faiss_index = None
vectorizer  = None


# ─────────────────────────────────────────
# Step 1 — Chunking
# ─────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of ~chunk_size words."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# ─────────────────────────────────────────
# Step 2 — Embeddings
# ─────────────────────────────────────────

def build_vectorizer_and_index(chunks: list[str]):
    """
    Fit TF-IDF on all chunks, L2-normalise, and load into FAISS.
    Inner product on unit vectors == cosine similarity.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(chunks).toarray().astype("float32")
    X = normalize(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return vec, index


# ─────────────────────────────────────────
# Step 3 — Retrieval
# ─────────────────────────────────────────

def retrieve_top_chunks(question: str, k: int = 4) -> list[dict]:
    """Embed question, search FAISS, return top-k chunks with scores."""
    q_vec = vectorizer.transform([question]).toarray().astype("float32")
    q_vec = normalize(q_vec)

    scores, indices = faiss_index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(text_chunks):
            results.append({
                "chunk": text_chunks[idx],
                "score": float(score),
            })
    return results


# ─────────────────────────────────────────
# Step 4 — Ollama answer generation
# ─────────────────────────────────────────

def generate_answer_with_groq(question: str, context: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Use the provided context to answer the question accurately. "
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n\n{context}\n\nQuestion:\n{question}"
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {e}"


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status":          "ok",
        "message":         "Mini NotebookLM backend (v6 — Ollama phi)",
        "document_loaded": faiss_index is not None,
        "num_chunks":      len(text_chunks),
        "llm":             f"Ollama / {OLLAMA_MODEL} @ {OLLAMA_URL}",
    })


@app.route("/upload", methods=["POST"])
def upload():
    global text_chunks, faiss_index, vectorizer

    if "file" not in request.files:
        return jsonify({"error": "No file found. Use form-data with key 'file'."}), 400

    pdf_file = request.files["file"]

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # ── Extract text from PDF ────────────────────────────────────────────
    try:
        reader     = pypdf.PdfReader(io.BytesIO(pdf_file.read()))
        pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
        full_text  = "\n".join(pages_text)
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {e}"}), 500

    if not full_text.strip():
        return jsonify({"error": "No readable text found (PDF may be image-only)."}), 422

    # ── Chunk → embed → index ────────────────────────────────────────────
    try:
        text_chunks             = split_into_chunks(full_text, chunk_size=500)
        vectorizer, faiss_index = build_vectorizer_and_index(text_chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to build index: {e}"}), 500

    return jsonify({
        "message":         "PDF uploaded and indexed successfully.",
        "filename":        pdf_file.filename,
        "chars_extracted": len(full_text),
        "num_chunks":      len(text_chunks),
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "'question' is required and cannot be empty."}), 400
    if faiss_index is None or not text_chunks:
        return jsonify({"error": "No document loaded. POST a PDF to /upload first."}), 400

    try:
        # 1. Retrieve the most relevant chunks via TF-IDF + FAISS
        top_results = retrieve_top_chunks(question, k=4)

        # 2. Combine chunks into a single context block for the LLM
        #    The "---" separator helps phi distinguish chunk boundaries
        context = "\n\n---\n\n".join(r["chunk"] for r in top_results)

        # 3. Generate a grounded answer with the local phi model
        answer = generate_answer_with_groq(question, context)

    except Exception as e:
        return jsonify({"error": f"RAG pipeline failed: {e}"}), 500

    return jsonify({
        "question":      question,
        "answer":        answer,
        "source_chunks": [r["chunk"] for r in top_results],
        "scores":        [r["score"] for r in top_results],
    })


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"✅  Ollama RAG server  |  model: {OLLAMA_MODEL}  |  url: {OLLAMA_URL}")
    print("    Before starting, make sure:")
    print("      1. Ollama is running  →  ollama serve")
    print("      2. phi is installed   →  ollama pull phi")
    app.run(host="0.0.0.0", port=5000, debug=True)