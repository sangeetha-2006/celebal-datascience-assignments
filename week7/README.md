# RAG Document Question Answering (Local, No API Key)

A minimal, fully local Retrieval-Augmented Generation pipeline for answering
questions about your own documents (.txt or .pdf).

## How it maps to the assignment architecture

| Stage | Implementation |
|---|---|
| 1. Document Ingestion | `load_document()` — reads .txt or .pdf |
| 2. Text Chunking | `chunk_text()` — overlapping character-based chunks |
| 3. Embedding Creation | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| 4. Vector Database | `FAISS` (`IndexFlatIP`, cosine similarity) |
| 5. Query Processing | Query embedded with the same embedding model |
| 6. Context Retrieval | `VectorStore.search()` — top-k similarity search |
| 7. Answer Generation | `google/flan-t5-base` via Hugging Face `transformers` |

## Setup (Local / VS Code / Jupyter)

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > First run will download the embedding model (~80MB) and the FLAN-T5
   > model (~250MB) from Hugging Face — this needs internet access once,
   > then everything runs offline.

3. Run the pipeline:
   ```bash
   python rag_pipeline.py
   ```
   It will load `sample_document.txt` and open an interactive prompt.
   Try asking:
   - "What is RAG?"
   - "What is FAISS used for?"
   - "What are the limitations of RAG?"

## Using your own document

Open `rag_pipeline.py` and change this line near the bottom:
```python
DOC_PATH = "sample_document.txt"
```
to point to your own file, e.g. `"my_resume.pdf"` or `"notes.txt"`.
Place the file in the same folder as the script (or give a full path).

## Using in Jupyter instead of a script

You can copy the classes (`VectorStore`, `AnswerGenerator`, `RAGPipeline`)
into a notebook cell, then run:
```python
rag = RAGPipeline()
rag.ingest("sample_document.txt")
answer, sources = rag.ask("What is the main idea of the document?")
print(answer)
```

## Suggested experiments (from the assignment's "Improvements" section)

- **Chunking**: try `chunk_size=300` vs `800`, or switch to sentence/paragraph
  based splitting instead of raw character slicing.
- **Embedding model**: swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (slower,
  more accurate) and compare retrieval quality.
- **Hybrid search**: add a simple keyword filter (e.g. TF-IDF or BM25) alongside
  vector search and combine the scores.
- **Re-ranking**: after retrieving top-10 chunks with FAISS, re-score them with
  a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) and keep
  the top 3.
- **Generation model**: try `google/flan-t5-large` (better quality, slower) if
  your machine can handle it.

## Notes

- Everything runs on CPU — no GPU or paid API required.
- `IndexFlatIP` does an exact (brute-force) search, which is fine for small
  documents. For very large document collections, look into FAISS's
  approximate indexes (e.g. `IndexIVFFlat`) for speed.
