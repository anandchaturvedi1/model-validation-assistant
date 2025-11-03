"""Utility to preprocess documents and build a FAISS vectorstore locally.
Usage:
    python scripts/preprocess_and_index.py path/to/doc.pdf --outdir data/index
"""
import argparse, os
from pathlib import Path
from app.rag import build_vectorstore_from_file

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input document path (.txt or .pdf)')
    p.add_argument('--outdir', default='data/index', help='Output dir for vectorstore')
    args = p.parse_args()
    vect = build_vectorstore_from_file(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    vect.save_local(str(outdir))
    print('Saved vectorstore to', outdir)

if __name__ == '__main__':
    main()
