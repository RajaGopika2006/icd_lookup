import os
import json
import pickle
from datetime import datetime
from typing import List

import streamlit as st
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import requests

st.set_page_config(
    page_title="ICD Code Finder Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Configuration -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_URL = os.getenv(
    "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
EMBS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------- Utilities -----------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


embed_model = load_embed_model()


def extract_pdf_chunks_bytes(pdf_bytes: bytes, max_chars: int = 1200):
    """Extract text from PDF into reasonably sized chunks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text() or ""
        text = text.replace("\r", "\n")

        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > max_chars:
                if current.strip():
                    out_chunks.append(
                        {"page": page_num + 1, "text": current.strip()}
                    )
                current = ""
            current += line + "\n"

        if current.strip():
            out_chunks.append({"page": page_num + 1, "text": current.strip()})

    return out_chunks


def build_embeddings_for_chunks(chunks_list: List[dict]):
    texts = [c["text"] for c in chunks_list]
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs


def save_chunks_and_embs(chunks_list, embs):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_list, f)
    np.save(EMBS_PATH, embs)


def load_chunks_and_embs():
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        embs = np.load(EMBS_PATH)
        return chunks, embs
    return None, None


def cosine_sim(a, b):
    return float(np.dot(a, b) / (max(1e-9, norm(a) * norm(b))))


def call_llm_system_prompt(system_prompt: str, user_prompt: str, max_tokens=400):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    resp = requests.post(LLM_API_URL, headers=headers, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM API error: {resp.status_code} {resp.text}")

    j = resp.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected LLM response format: " + str(e))


# ----------------------- Corrections store -----------------------
def load_corrections():
    if os.path.exists(CORRECTIONS_PATH):
        with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_corrections(corrs):
    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrs, f, ensure_ascii=False, indent=2)


def add_correction(query, model_icd, correct_icd, notes="", embed=None):
    corrs = load_corrections()
    rec = {
        "id": len(corrs) + 1,
        "query": query,
        "original_icd": model_icd,
        "correct_icd": correct_icd,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(),
        "vector": embed.tolist() if embed is not None else None,
    }
    corrs.append(rec)
    save_corrections(corrs)


def find_similar_correction(query, top_k=1):
    corrs = load_corrections()
    if not corrs:
        return None, 0.0

    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    best = None
    best_sim = -1.0

    for c in corrs:
        if c.get("vector") is None:
            continue
        v = np.array(c["vector"], dtype=float)
        sim = cosine_sim(q_vec, v)
        if sim > best_sim:
            best_sim = sim
            best = c

    return best, best_sim


# ----------------------- Query pipeline -----------------------
def get_top_chunks_for_query(query, chunks, embeddings, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    sims = [(cosine_sim(q_emb, embeddings[i]), i) for i in range(len(chunks))]
    sims.sort(reverse=True, key=lambda x: x[0])
    top = sims[:top_k]
    selected = [chunks[i] for _, i in top]
    return selected, q_emb, [s for s, _ in top]


def query_icd_from_pdf(query, top_k=5):
    chunks, embeddings = load_chunks_and_embs()
    if not chunks or embeddings is None:
        return {"error": "No PDF indexed. Upload a PDF first."}

    # 1) Check corrections first
    corr, sim = find_similar_correction(query)
    if corr and sim >= 0.92:
        return {
            "icd_code": corr["correct_icd"],
            "disease": None,
            "page": None,
            "snippet": None,
            "reason": f"Using previously corrected answer (similarity={sim:.3f})",
            "from_correction": True,
            "correction_id": corr["id"],
        }

    # 2) RAG over PDF chunks
    selected_chunks, q_vec, sims = get_top_chunks_for_query(
        query, chunks, embeddings, top_k=top_k
    )

    chunks_text = ""
    for c in selected_chunks:
        chunks_text += f"[Page {c['page']}]\n{c['text']}\n\n"

    system_prompt = (
        "You are an assistant whose only job is to find explicit ICD codes "
        "inside the provided document sections. "
        "You MUST NOT invent or hallucinate ICD codes. "
        "You MUST only return codes that appear verbatim in the text. "
        "If multiple ICD codes appear, return the one that best matches the "
        "user symptoms, and include the exact snippet and page number."
    )

    user_prompt = (
        f"User symptoms / query:\n{query}\n\n"
        f"Document snippets (ONLY these):\n{chunks_text}\n"
        "Task:\n"
        "1) Find an explicit ICD code (for example 'ICD-10 A15.0' or 'A15.0') "
        "that appears in the text above and best matches the user query.\n"
        "2) If you find a code, output a JSON object with these keys exactly: "
        '{"icd_code": string or null, "disease": string or null, '
        '"page": int or null, "snippet": string or null, "reason": string or null}.\n'
        '3) If you do NOT find an explicit ICD code in the supplied snippets, return '
        '{"icd_code": null, "reason": "No ICD code found in the provided document sections."}.\n'
        "4) Do NOT provide any other text besides the JSON object.\n"
    )

    try:
        raw = call_llm_system_prompt(system_prompt, user_prompt, max_tokens=400)
    except Exception as e:
        return {"error": f"LLM error: {e}"}

    # Parse JSON
    try:
        parsed = json.loads(raw.strip())
    except Exception:
        return {
            "error": "LLM output could not be parsed as JSON.",
            "raw": raw,
        }

    icd_code = parsed.get("icd_code")
    snippet = parsed.get("snippet")
    page = parsed.get("page")
    disease = parsed.get("disease")
    reason = parsed.get("reason")

    # Validate icd_code appears in selected chunks
    if icd_code:
        found = False
        for c in selected_chunks:
            if icd_code in c["text"]:
                found = True
                break
        if not found:
            return {
                "error": "LLM returned a code not present in the retrieved PDF snippets.",
                "raw": raw,
            }

    return {
        "icd_code": icd_code,
        "disease": disease,
        "page": page,
        "snippet": snippet,
        "reason": reason,
        "from_correction": False,
        "search_scores": sims,
    }


# ------------------------------- MODERN UI START -------------------------------

# Enhanced CSS styling with gradient backgrounds and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Custom header */
    .main-header {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #667eea;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
    }
    
    .result-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
    }
    
    .result-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .result-item strong {
        color: #fff;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stats-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #10b981;
        color: white;
    }
    
    .badge-warning {
        background: #f59e0b;
        color: white;
    }
    
    .badge-info {
        background: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "show_correction_form" not in st.session_state:
    st.session_state["show_correction_form"] = False


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🩺 ICD Code Finder")
    st.markdown("---")
    
    # PDF Status
    st.markdown("#### 📚 PDF Index Status")
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBS_PATH):
        st.success("✅ PDF Indexed & Ready")
        chunks, _ = load_chunks_and_embs()
        if chunks:
            st.metric("Total Chunks", len(chunks))
    else:
        st.warning("⚠ No PDF indexed")
        st.info("Click below to index the built-in medical PDF")
        
        if st.button("🚀 Index PDF Now", key="index_btn"):
            with st.spinner("Indexing PDF..."):
                try:
                    with open("data/icd_source.pdf", "rb") as f:
                        data = f.read()
                    chunks = extract_pdf_chunks_bytes(data)
                    embs = build_embeddings_for_chunks(chunks)
                    save_chunks_and_embs(chunks, embs)
                    st.success("✅ PDF Indexed Successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Corrections Stats
    st.markdown("#### 📝 Corrections Database")
    corrections = load_corrections()
    st.metric("Total Corrections", len(corrections))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 View All", key="view_corr"):
            st.session_state["show_corrections"] = True
    with col2:
        if st.button("🗑 Clear", key="clear_corr"):
            if os.path.exists(CORRECTIONS_PATH):
                os.remove(CORRECTIONS_PATH)
                st.success("Cleared!")
                st.rerun()
    
    st.markdown("---")
    
    # System Info
    st.markdown("#### ⚙ System Status")
    st.markdown(f"*API Key:* {'🟢 Connected' if OPENAI_API_KEY else '🔴 Missing'}")
    st.markdown(f"*Embedding Model:* {EMBED_MODEL_NAME}")
    
    st.markdown("---")
    
    if st.button("🔄 Reset All Data", key="reset_all"):
        if os.path.exists(CHUNKS_PATH): os.remove(CHUNKS_PATH)
        if os.path.exists(EMBS_PATH): os.remove(EMBS_PATH)
        if os.path.exists(CORRECTIONS_PATH): os.remove(CORRECTIONS_PATH)
        st.success("All data reset!")
        st.rerun()


# ---------------- Main Page ----------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 ICD Code Finder Assistant</h1>
    <p>Intelligent medical code lookup </p>
</div>
""", unsafe_allow_html=True)



st.markdown("<br>", unsafe_allow_html=True)

# Search Section

st.markdown("### 🔍 Enter Patient Symptoms")
st.markdown("Describe the symptoms clearly and specifically for best results")

query = st.text_area(
    "Symptoms Description",
    placeholder="Example: Patient presents with persistent dry cough lasting 3 weeks, low-grade fever (38°C), night sweats, and unexplained weight loss of 5kg...",
    height=150,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    top_k = st.slider("Search Depth", 1, 10, 5, help="Number of document chunks to analyze")

with col3:
    search_btn = st.button("🔎 Find ICD Code", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Search Logic
if search_btn:
    if not query.strip():
        st.warning("⚠ Please enter symptoms first!")
    else:
        with st.spinner("🔄 Analyzing symptoms and searching medical database..."):
            res = query_icd_from_pdf(query, top_k=top_k)
            st.session_state["last_query"] = query
            st.session_state["last_result"] = res

# Display Results
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    
    if res.get("error"):
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.error(f"❌ Error: {res['error']}")
        if "raw" in res:
            with st.expander("🔍 View Raw Response"):
                st.code(res["raw"][:1000], language="text")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Success Result
        st.markdown(f"""
        <div class="result-card">
            <h3>✅ ICD Code Found</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.markdown("#### 📋 Code Details")
            
            if res.get("icd_code"):
                st.markdown(f"*ICD Code:* {res['icd_code']}")
            else:
                st.markdown("*ICD Code:* Not found")
            
            if res.get("disease"):
                st.markdown(f"*Disease:* {res['disease']}")
            
            if res.get("page"):
                st.markdown(f"*Source Page:* {res['page']}")
            
            if res.get("from_correction"):
                st.markdown('<span class="badge badge-info">📝 From Correction Database</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-success">🔍 From PDF Search</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        
        
        # Feedback Section
        
        st.markdown("### 💬 Is this result accurate?")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Yes, Correct", use_container_width=True):
                st.success("Thank you for confirming! This helps improve our system.")
        
        with col2:
            if st.button("❌ No, Incorrect", use_container_width=True):
                st.session_state["show_correction_form"] = True
        
        st.markdown('</div>', unsafe_allow_html=True)

# Correction Form
if st.session_state.get("show_correction_form"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### ✏ Submit Correction")
    st.markdown("Help us improve by providing the correct ICD code")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_icd = st.text_input("Correct ICD Code", placeholder="e.g., A15.0")
    
    with col2:
        corr_notes = st.text_area("Additional Notes (Optional)", placeholder="Any additional context...", height=100)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("💾 Save Correction", type="primary", use_container_width=True):
            if corr_icd.strip():
                last_query = st.session_state["last_query"]
                q_emb = embed_model.encode([last_query], convert_to_numpy=True)[0]
                add_correction(
                    last_query,
                    res.get("icd_code"),
                    corr_icd,
                    notes=corr_notes,
                    embed=q_emb,
                )
                st.success("✅ Correction saved! Future searches will be improved.")
                st.session_state["show_correction_form"] = False
                st.rerun()
            else:
                st.warning("Please enter a valid ICD code")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state["show_correction_form"] = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# View Corrections Modal
if st.session_state.get("show_corrections"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Saved Corrections")
    
    corrections = load_corrections()
    if corrections:
        for corr in reversed(corrections[-10:]):  # Show last 10
            with st.expander(f"#{corr['id']} - {corr['correct_icd']}"):
                st.markdown(f"*Query:* {corr['query']}")
                st.markdown(f"*Original:* {corr['original_icd']} → *Corrected:* {corr['correct_icd']}")
                if corr.get('notes'):
                    st.markdown(f"*Notes:* {corr['notes']}")
                st.caption(f"Saved: {corr['timestamp'][:10]}")
    else:
        st.info("No corrections saved yet")
    
    if st.button("Close", use_container_width=True):
        st.session_state["show_corrections"] = False
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p style="margin: 0; font-size: 0.9rem;">🩺 ICD Code Finder Assistant</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">Powered by Advanced AI • PDF Indexing • Semantic Search</p>
</div>
""", unsafe_allow_html=True)