# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil, uuid, os, json

# RAG helper
from rag import build_vectorstore_from_file, demo_qa

# LangChain/embeddings used to reload saved FAISS stores
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = FastAPI(title="Model Validation Assistant - Demo")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
VECTOR_DIR = DATA_DIR / "vectorstores"
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True, parents=True)

# default embedding model used for saving and loading vectorstores
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@app.get('/health')
def health():
    return {"status":"ok"}

@app.post('/upload_doc')
async def upload_doc(file: UploadFile = File(...)):
    """
    Saves uploaded file and builds + persists a FAISS vectorstore for it.
    Returns a doc_id (UID) which can be used with /ask to query that doc.
    """
    # Save uploaded file to data directory
    uid = uuid.uuid4().hex[:8]
    safe_name = file.filename.replace(' ', '_')
    out_path = DATA_DIR / f"{uid}_{safe_name}"
    try:
        with open(out_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Build vectorstore from file and persist it
    try:
        vect = build_vectorstore_from_file(str(out_path), embedding_model=DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        # If embedding/build fails, still return file saved but signal failure
        return JSONResponse({"filename": str(out_path), "doc_id": None, "error": f"failed to build vectorstore: {e}"}, status_code=500)

    vs_dir = VECTOR_DIR / uid
    vs_dir.mkdir(parents=True, exist_ok=True)
    # persist the FAISS index locally
    try:
        vect.save_local(str(vs_dir))
    except Exception as e:
        return JSONResponse({"filename": str(out_path), "doc_id": None, "error": f"failed to save vectorstore: {e}"}, status_code=500)

    # Write metadata for convenience
    meta = {
        "original_filename": safe_name,
        "source_path": str(out_path),
        "embedding_model": DEFAULT_EMBEDDING_MODEL
    }
    with open(vs_dir / "meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf)

    return {"filename": str(out_path), "doc_id": uid}

@app.post('/ask')
async def ask(question: str = Form(...), doc_id: str = Form(None), doc_filename: str = Form(None)):
    """
    Ask a question. Prefer doc_id (returned by /upload_doc).
    If doc_id is provided, this will load the persisted FAISS store and run the RAG pipeline.
    If doc_filename is provided and exists (path under data/), it will build a temporary vectorstore from that file (demo mode).
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    vectorstore = None
    # 1) Prefer doc_id -> load persisted vectorstore
    if doc_id:
        vs_dir = VECTOR_DIR / doc_id
        if not vs_dir.exists():
            raise HTTPException(status_code=404, detail=f"Vectorstore for doc_id '{doc_id}' not found.")
        # load embeddings with same model used to save
        hf = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
        try:
            vectorstore = FAISS.load_local(str(vs_dir), hf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load vectorstore: {e}")

    # 2) Fallback: if doc_filename provided and points to an existing file on disk, build ephemeral vectorstore
    elif doc_filename:
        p = Path(doc_filename)
        # If user sent relative or just filename that actually sits in DATA_DIR, resolve it
        if not p.exists():
            alt = DATA_DIR / doc_filename
            if alt.exists():
                p = alt
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Document file '{doc_filename}' not found.")
        try:
            vectorstore = build_vectorstore_from_file(str(p), embedding_model=DEFAULT_EMBEDDING_MODEL)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build vectorstore from file: {e}")
    else:
        # No document provided
        return JSONResponse({
            "question": question,
            "doc_provided": False,
            "answer": "No document provided. Please upload a document and pass its doc_id (or provide doc_filename)."
        }, status_code=400)

    # Run the RAG QA chain using the helper from rag.py
    try:       
        answer_text = demo_qa(vectorstore, question, llm_name='openai')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {e}")

    return JSONResponse({
        "question": question,
        "doc_provided": True,
        "doc_id": doc_id,
        "answer": answer_text
    })