# icd_backend/icd_engine.py

import os
import io
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Optional

import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import requests

# ----------------------- Configuration -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

DATA_DIR = os.getenv("ICD_DATA_DIR", "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
EMBS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------- Load embed model -----------------------
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def get_embed_model():
    return _embed_model


# ----------------------- PDF handling -----------------------
def extract_pdf_chunks_bytes(pdf_bytes: bytes, max_chars: int = 1200) -> List[dict]:
    """
    Extract text chunks from PDF (bytes) with small-ish segments per page.
    """
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
                    out_chunks.append({"page": page_num + 1, "text": current.strip()})
                current = ""
            current += line + "\n"
        if current.strip():
            out_chunks.append({"page": page_num + 1, "text": current.strip()})
    return out_chunks


def build_embeddings_for_chunks(chunks_list: List[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks_list]
    embs = get_embed_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs


def save_chunks_and_embs(chunks_list: List[dict], embs: np.ndarray) -> None:
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_list, f)
    np.save(EMBS_PATH, embs)


def load_chunks_and_embs() -> Tuple[Optional[List[dict]], Optional[np.ndarray]]:
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        embs = np.load(EMBS_PATH)
        return chunks, embs
    return None, None


# ----------------------- Similarity & LLM -----------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (max(1e-9, norm(a) * norm(b))))


def call_llm_system_prompt(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
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
    corrs.append(rec    )
    save_corrections(corrs)


def find_similar_correction(query, threshold: float = 0.92):
    corrs = load_corrections()
    if not corrs:
        return None, 0.0
    q_vec = get_embed_model().encode([query], convert_to_numpy=True)[0]
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
    if best and best_sim >= threshold:
        return best, best_sim
    return None, best_sim


# ----------------------- RAG pipeline -----------------------
def get_top_chunks_for_query(query: str, chunks, embeddings, top_k: int = 5):
    q_emb = get_embed_model().encode([query], convert_to_numpy=True)[0]
    sims = [(cosine_sim(q_emb, embeddings[i]), i) for i in range(len(chunks))]
    sims.sort(reverse=True, key=lambda x: x[0])
    top = sims[:top_k]
    selected = [chunks[i] for _, i in top]
    return selected, q_emb, [s for s, _ in top]


def query_icd_from_pdf(query: str, top_k: int = 5) -> dict:
    chunks, embeddings = load_chunks_and_embs()
    if not chunks or embeddings is None:
        return {"error": "No PDF indexed. Upload a PDF first."}

    # 1. Check corrections first
    corr, sim = find_similar_correction(query)
    if corr:
        return {
            "icd_code": corr["correct_icd"],
            "disease": None,
            "page": None,
            "snippet": None,
            "reason": f"Using previously corrected answer (similarity={sim:.3f})",
            "from_correction": True,
            "correction_id": corr["id"],
        }

    # 2. RAG over PDF chunks
    selected_chunks, q_vec, sims = get_top_chunks_for_query(
        query, chunks, embeddings, top_k=top_k
    )
    chunks_text = ""
    for c in selected_chunks:
        chunks_text += f"[Page {c['page']}]\n{c['text']}\n\n"

    system_prompt = (
        "You are an assistant whose only job is to *find explicit ICD codes* "
        "inside the provided document sections. "
        "You MUST NOT invent or hallucinate ICD codes. You MUST only return codes "
        "that appear verbatim in the text. "
        "If multiple ICD codes appear, return the one that best matches the user symptoms, "
        "and include the exact snippet and page number."
    )

    user_prompt = (
        f"User symptoms / query:\n{query}\n\n"
        f"Document snippets (ONLY these):\n{chunks_text}\n"
        "Task:\n"
        "1) Find an explicit ICD code (for example 'ICD-10 A15.0' or 'A15.0') that appears "
        "in the text above and best matches the user query.\n"
        "2) If you find a code, output a JSON object with these keys exactly: "
        '{"icd_code": string or null, "disease": string or null, "page": int or null, '
        '"snippet": string or null, "reason": string or null}.\n'
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
        return {"error": "LLM output could not be parsed as JSON.", "raw": raw}

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
