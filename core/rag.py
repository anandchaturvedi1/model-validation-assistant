"""RAG utilities for Model Validation Assistant
This version removes dependency on langchain_community document loaders.
Instead, it parses files directly to text using built-in Python and PyPDF2/docx.
"""

import os
from pathlib import Path
import difflib
from typing import List

#import PyPDF2
#import docx
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
'''


import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline


def extract_text_from_file(file_path: str, dummy = 1) -> str:
    """Extract text from a txt, pdf, or docx file using Python libraries."""
    suffix = Path(file_path).suffix.lower()
    text = ""

    if suffix == '.pdf':
		    pass
        # with open(file_path, 'rb') as f:
            #reader = PyPDF2.PdfReader(f)
            #for page in reader.pages:
                #text += page.extract_text() or ""
    elif suffix in ('.docx', '.doc'):
		    pass
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    return [text.strip(), "dummy text 1", "dummy text 2"] if dummy == 1 else text.strip()


def build_vectorstore_from_file(path: str, embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 100):
    """Create FAISS vectorstore from a file (txt/pdf/docx) by extracting text and splitting into chunks."""
    documents = extract_text_from_file(path)
    print("documents: ", documents)
    # --- Step 2: Embed the documents ---
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    doc_embeddings = embedding_model.embed_documents(documents)

    # --- Step 3: Build FAISS index ---
    dimension = len(doc_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    # Map FAISS indices to documents
    id_to_doc = {i: documents[i] for i in range(len(documents))}
    return index , id_to_doc, embedding_model


# --- Step 4: Define a retriever function ---
def retrieve(query, index, id_to_doc, embedding_model, top_k=2):
    """Return top_k most similar documents for a query"""
    query_embedding = embedding_model.embed_query(query)
    D, I = index.search(np.array([query_embedding]), top_k)
    return [id_to_doc[i] for i in I[0]]
	



def create_local_hf_llm(model_name: str = 'gpt2', max_new_tokens: int = 128, temperature: float = 0.7):
	# --- Step 5: Load the LLM ---
	model_name = "google/flan-t5-base"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	hf_pipeline = pipeline(
		"text2text-generation",
		model=model,
		tokenizer=tokenizer,
		max_new_tokens=150
	)
	llm = HuggingFacePipeline(pipeline=hf_pipeline)
	return llm
	'''
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    pipe = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)
	'''


#def demo_qa(vectorstore: FAISS, question: str, top_k: int = 4, hf_model: str = 'gpt2') -> str:
# --- Step 6: Define manual RetrievalQA function ---
def demo_qa(query: str,  file_path, top_k=2):
  index , id_to_doc, embedding_model = build_vectorstore_from_file(file_path)
  relevant_docs = retrieve(query, index , id_to_doc, embedding_model, top_k=top_k)
  context = "\n".join(relevant_docs)
  prompt = f"Answer the following question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
  llm = create_local_hf_llm()
  # using invoke() instead of calling the object directly
  response = llm.invoke(prompt)

  # HuggingFacePipeline returns a string directly
  return response.strip()
	
def summarize_model_doc(file_path: str, hf_model: str = 'gpt2') -> str:
    # Build vector store
    index, id_to_doc, embedding_model = build_vectorstore_from_file(file_path)
    llm = create_local_hf_llm(model_name="google/flan-t5-base")  # better summarizer

    # Define actual query
    query = "Summarize the model document clearly under headings: Assumptions, Inputs, and Results."

    # Retrieve relevant content
    relevant_docs = retrieve(query, index, id_to_doc, embedding_model, top_k=2)
    context = "\n".join(relevant_docs)

    # Construct prompt
    prompt = (
        f"Context:\n{context}\n\n"
        "Summarize this model description under these headings:\n"
        "# Assumptions\n"
        "# Inputs\n"
        "# Results\n"
        "Answer:"
    )
    print("prompt : " , prompt)
    # Run model
    response = llm.invoke(prompt)
    return response.strip()


def compare_model_versions(file_a: str, file_b: str) -> str:
    text_a = extract_text_from_file(file_a,0).splitlines(keepends=True)
    text_b = extract_text_from_file(file_b,0).splitlines(keepends=True)
    diff = difflib.unified_diff(text_a, text_b, fromfile=file_a, tofile=file_b, n=3)
    result = "".join(diff)
    return result or "No significant differences found."