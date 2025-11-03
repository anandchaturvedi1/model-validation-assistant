"""
Gradio app for Hugging Face Space. Uses core_app.rag utilities.
"""
import gradio as gr
from pathlib import Path
import difflib
import os
import shutil

# Import the RAG utilities from the core_app package
from core.rag import build_vectorstore_from_file, demo_qa, summarize_model_doc, compare_model_versions


# ---------- Gradio handlers ----------

def run_qa_old(file, question):
    if file is None or not question or not question.strip():
        return "Please upload a document and enter a question."
    tmp_dir = Path('/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.name
    with open(tmp_path, 'wb') as f:
        f.write(file.read())
    try:
        vect = build_vectorstore_from_file(str(tmp_path))
        answer = demo_qa(question, tmp_path)
        return answer
    except Exception as e:
        return f"Error: {e}"

def run_qa(file, question):
    if file is None or not question or not question.strip():
        return "Please upload a document and enter a question."

    tmp_dir = Path('/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / Path(file.name).name

    try:
        # copy the uploaded file from Gradio temp to our /tmp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        shutil.copy(file, tmp_path)
        answer = demo_qa(question, tmp_path)
        return answer
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return f"Error:\n{tb}"
        
def run_summary_old(file):
    if file is None:
        return "Please upload a document to summarize."
    tmp_dir = Path('/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.name 
    if os.path.exists(tmp_path):
            os.remove(tmp_path)
    shutil.copy(file, tmp_path)
    try:
        return summarize_model_doc(str(tmp_path))
    except Exception as e:
        return f"Error during summarization: {e}"

def run_summary(file):
    if file is None:
        return "Please upload a document to summarize."
    try:
        # file is already a valid path string (e.g. /tmp/gradio/abc123.txt)
        print(f"Received file path: {file}")
        return summarize_model_doc(str(file))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return f"Error during summarization:\n{tb}"

def run_comparison_old(file1, file2):
    if file1 is None or file2 is None:
        return "Please upload both model versions."
    tmp_dir = Path('/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p1 = tmp_dir / file1.name
    p2 = tmp_dir / file2.name
    if os.path.exists(p1):
            os.remove(p1)
    shutil.copy(file1, p1)
    if os.path.exists(p1):
            os.remove(p1)
    shutil.copy(file2, p2)
    try:
        return compare_model_versions(str(p1), str(p2))
    except Exception as e:
        return f"Error during comparison: {e}"

def run_comparison(file1, file2):
    if file1 is None or file2 is None:
        return "Please upload both model versions."
    try:
        return compare_model_versions(str(file1), str(file2))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return f"Error during comparison:\n{tb}"

# ---------- Gradio UI ----------
with gr.Blocks(title="Model Validation Assistant (RAG + Comparison)") as demo:
    gr.Markdown(
        """
        # 🧠 Model Validation Assistant  
        An AI copilot to help model validators and risk modelers review, explain, and document financial models.
        **Powered by LangChain and Retrieval-Augmented Generation (RAG).**
        """
    )

    with gr.Tab("Ask / QA"):
        gr.Markdown("Upload a model document and ask validation-related questions.")
        with gr.Row():
            doc = gr.File(label="Upload document (.pdf, .txt, .docx)")
            qbox = gr.Textbox(label="Question", placeholder="e.g. What are the main assumptions?")
        ans_box = gr.Textbox(label="Answer", lines=8)
        gr.Button("Ask").click(run_qa, [doc, qbox], ans_box)

    with gr.Tab("Summarize Model"):
        gr.Markdown("Summarize the model’s assumptions, inputs, and results automatically.")
        sum_doc = gr.File(label="Upload document")
        sum_out = gr.Textbox(label="Summary", lines=12)
        gr.Button("Summarize").click(run_summary, sum_doc, sum_out)

    with gr.Tab("Compare Versions"):
        gr.Markdown("Compare two model documentation files and highlight validation-relevant differences.")
        with gr.Row():
            old_doc = gr.File(label="Old version")
            new_doc = gr.File(label="New version")
        diff_out = gr.Textbox(label="Differences", lines=15)
        gr.Button("Compare").click(run_comparison, [old_doc, new_doc], diff_out)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True
    )