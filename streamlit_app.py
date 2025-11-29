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
    page_title="PDF → ICD Chatbot (Streamlit)",
    layout="wide"
)

# ----------------------- Configuration -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set this in your environment
st.sidebar.write("DEBUG - API key present:", bool(OPENAI_API_KEY))
LLM_API_URL = os.getenv(
    "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
)
st.sidebar.write("DEBUG - LLM URL:", LLM_API_URL)
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
        "You are an assistant whose only job is to *find explicit ICD codes* "
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




# Sidebar: PDF + corrections
with st.sidebar:
    st.header("Index PDF")
    uploaded = st.file_uploader(
        "Upload Indian medical signs PDF", type=["pdf"]
    )
    if uploaded is not None:
        if st.button("Index uploaded PDF"):
            data = uploaded.read()
            with st.spinner("Extracting and indexing..."):
                local_chunks = extract_pdf_chunks_bytes(data)
                if not local_chunks:
                    st.error("No text found in PDF")
                else:
                    embs = build_embeddings_for_chunks(local_chunks)
                    save_chunks_and_embs(local_chunks, embs)
                    st.success(f"Indexed {len(local_chunks)} chunks")

    if st.button("Clear indexed PDF"):
        try:
            if os.path.exists(CHUNKS_PATH):
                os.remove(CHUNKS_PATH)
            if os.path.exists(EMBS_PATH):
                os.remove(EMBS_PATH)
            st.success("Cleared index")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.header("Corrections")

    if st.button("Show corrections"):
        corrs = load_corrections()
        st.write(corrs)

    if st.button("Clear corrections"):
        if os.path.exists(CORRECTIONS_PATH):
            os.remove(CORRECTIONS_PATH)
            st.success("Cleared corrections")


# Main area: query + answer
st.subheader("Ask about symptoms / disease")
query = st.text_area(
    "Enter symptoms (e.g. fever, night sweats, chronic cough)", height=120
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.number_input(
        "Chunks to search (top_k)", min_value=1, max_value=10, value=5
    )
with col2:
    if st.button("Get ICD"):
        if not query.strip():
            st.warning("Type some symptoms or disease description")
        else:
            with st.spinner("Searching PDF and consulting LLM..."):
                res = query_icd_from_pdf(query, top_k=top_k)
                st.session_state["last_query"] = query
                st.session_state["last_result"] = res
                st.rerun()

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    st.markdown("---")

    if res.get("error"):
        st.error(res["error"])
        if "raw" in res:
            st.code(res["raw"][:1000])
    else:
        st.success("Result")
        st.json(res)

        st.markdown("**Is this correct?**")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("✅ Yes — correct"):
                st.success("Thanks for confirming!")
        with c2:
            if st.button("❌ No — wrong"):
                st.session_state["show_correction_form"] = True

    if st.session_state.get("show_correction_form"):
        st.markdown("### Submit correction")
        corr_icd = st.text_input("Correct ICD code (e.g. A16.2)")
        corr_notes = st.text_area("Optional notes")
        if st.button("Submit correction"):
            last_query = st.session_state.get("last_query", "")
            q_emb = embed_model.encode(
                [last_query], convert_to_numpy=True
            )[0]
            add_correction(
                last_query,
                res.get("icd_code"),
                corr_icd,
                notes=corr_notes,
                embed=q_emb,
            )
            st.success(
                "Correction saved. The system will prefer this answer for similar queries."
            )
            st.session_state["show_correction_form"] = False

st.markdown("---")
st.write(
    "Notes: This app expects the uploaded PDF to contain explicit ICD codes "
    "in text form. Corrections are stored locally in the `data` folder."
)
