# RAG Document Question Answering (Local, No API Key)

A minimal, fully local Retrieval-Augmented Generation pipeline for answering
questions about your own documents (.txt or .pdf).

## How it maps to the assignment architecture

| Stage | Implementation |
|---|---|
| 1. Document Ingestion | `load_document()` — reads .txt/.pdf; `load_from_huggingface()` — streams a HF dataset |
| 2. Text Chunking | `chunk_text()` — overlapping character-based chunks |
| 3. Embedding Creation | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| 4. Vector Database | `FAISS` (`IndexFlatIP`, cosine similarity) |
| 5. Query Processing | Query embedded with the same embedding model |
| 6. Context Retrieval | `VectorStore.search()` — top-k similarity search, optional BM25 hybrid + cross-encoder re-ranking |
| 7. Answer Generation | `google/flan-t5-base` via `AutoModelForSeq2SeqLM` |
| 8. Optimization experiments | Chunk-size sweep, hybrid search, re-ranking — see `metrics_report.md` |

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

## Ingesting a Hugging Face dataset

Instead of a local file, you can pull text straight from a Hugging Face dataset
(e.g. a domain-specific archive like `vectara/open_ragbench`):

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_huggingface("vectara/open_ragbench", split="train", text_column="text", num_rows=50)

answer, sources = rag.ask("What is this dataset about?")
print(answer)
```

> Check the dataset's page on the Hugging Face Hub to confirm the correct
> `text_column` name — schemas vary between datasets.

## Hybrid search & re-ranking (Requirement 8 — optimization experiments)

Both are implemented and can be toggled on:

```python
rag = RAGPipeline(
    use_hybrid=True,                                       # BM25 + vector combined search
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder re-ranking
)
```

Or edit the `USE_HYBRID` / `RERANK_MODEL` variables at the top of the
`__main__` block in `rag_pipeline.py`.

- **Hybrid search** combines a normalized FAISS cosine-similarity score with
  a normalized BM25 keyword score (50/50 weighting) — helps when a query
  contains specific keywords/names the embedding model may under-weight.
- **Re-ranking** over-fetches the top 10 candidates from vector/hybrid search,
  then re-scores each (query, chunk) pair directly with a cross-encoder model,
  keeping only the true top-k. More accurate, but slower per query.

## Validation logs

Run the validation script to test a fixed set of sample questions and produce
a Markdown log of every question, its retrieved context, and its generated
answer:

```bash
python validate.py
```

This writes `validation_log.md`. Edit the `TEST_QUESTIONS` list inside
`validate.py` to match your own document's content.

## System metrics report

See [`metrics_report.md`](metrics_report.md) for full documentation of the
chunking profile, embedding model + dimensions, vector store configuration,
LLM setup, and a log of the optimization experiments (chunk size, hybrid
search, re-ranking) run against this pipeline.

## Suggested further experiments

- **Chunking**: try `chunk_size=300` vs `800`, or switch to sentence/paragraph
  based splitting instead of raw character slicing.
- **Embedding model**: swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (slower,
  more accurate) and compare retrieval quality.
- **Generation model**: try `google/flan-t5-large` (better quality, slower) if
  your machine can handle it.

## Notes

- Everything runs on CPU — no GPU or paid API required.
- `IndexFlatIP` does an exact (brute-force) search, which is fine for small
  documents. For very large document collections, look into FAISS's
  approximate indexes (e.g. `IndexIVFFlat`) for speed.
