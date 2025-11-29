# icd_backend/main.py

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from icd_engine import (
    extract_pdf_chunks_bytes,
    build_embeddings_for_chunks,
    save_chunks_and_embs,
    query_icd_from_pdf,
    add_correction,
    get_embed_model,
)

app = FastAPI(
    title="ICD PDF Chatbot API",
    description="Backend API for querying ICD codes from an indexed medical PDF",
    version="1.0.0",
)

# Allow CORS for frontend/mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class FeedbackRequest(BaseModel):
    query: str
    model_icd: Optional[str] = None
    correct_icd: str
    notes: Optional[str] = ""


@app.get("/")
def root():
    return {"message": "ICD PDF Chatbot API is running"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and index the Indian medical PDF.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")

    data = await file.read()
    chunks = extract_pdf_chunks_bytes(data)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    embs = build_embeddings_for_chunks(chunks)
    save_chunks_and_embs(chunks, embs)

    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.post("/query")
def query_icd(req: QueryRequest):
    """
    Query ICD code based on symptoms / disease description.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    res = query_icd_from_pdf(req.query, top_k=req.top_k)
    return res


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Submit a correction when the model gives a wrong ICD code.
    """
    if not req.correct_icd.strip():
        raise HTTPException(status_code=400, detail="correct_icd must not be empty")

    query_text = req.query or ""
    embed_model = get_embed_model()
    q_emb = embed_model.encode([query_text], convert_to_numpy=True)[0]

    add_correction(
        query=query_text,
        model_icd=req.model_icd,
        correct_icd=req.correct_icd,
        notes=req.notes or "",
        embed=q_emb,
    )

    return {"status": "saved"}
