# System Metrics Report — RAG Document Q&A Pipeline

## 1. Document Ingestion

| Property | Value |
|---|---|
| Supported input types | `.txt`, `.pdf`, Hugging Face datasets (streamed) |
| PDF parser | `pypdf.PdfReader` |
| HF dataset loader | `datasets.load_dataset(..., streaming=True)` |
| Preprocessing | Whitespace normalization (collapses newlines/tabs/multiple spaces) |

## 2. Chunking Profile

| Property | Value |
|---|---|
| Method | Fixed-size character window with overlap |
| Chunk size (default) | 500 characters |
| Overlap (default) | 50 characters |
| Rationale | Overlap prevents relevant sentences from being split across two chunks and losing retrievability |
| Sample document result | 8 chunks from ~3,600-character sample document |

> Tunable in `RAGPipeline(chunk_size=..., overlap=...)`. Larger chunk_size
> gives the LLM more context per retrieved chunk but reduces retrieval
> precision; smaller chunks are more precise but may lose surrounding context.

## 3. Embedding Model

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Similarity metric | Cosine similarity (implemented as inner product on L2-normalized vectors) |
| Why this model | Small (~80MB), fast on CPU, strong general-purpose semantic performance for its size — good fit for a local, no-GPU pipeline |

## 4. Vector Store

| Property | Value |
|---|---|
| Library | FAISS (`faiss-cpu`) |
| Index type | `IndexFlatIP` (exact brute-force inner-product search) |
| Storage | In-memory (rebuilt on each run; not persisted to disk) |
| Scalability note | `IndexFlatIP` is exact but O(n) per query — fine for small/medium document sets. For larger corpora, an approximate index such as `IndexIVFFlat` or `IndexHNSWFlat` would trade a small amount of accuracy for much faster search. |

## 5. Retrieval Configuration

| Property | Value |
|---|---|
| Default mode | Pure vector (cosine) similarity search |
| Optional hybrid mode | Combines normalized vector score + BM25 keyword score (50/50 weighting), via `rank_bm25.BM25Okapi` |
| Optional re-ranking | Cross-encoder re-scoring of over-fetched candidates (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`), keeping the true top-k after re-ranking |
| Default top-k | 3 chunks passed to the generator |

## 6. Language Model (Generation)

| Property | Value |
|---|---|
| Model | `google/flan-t5-base` |
| Architecture | Seq2Seq (encoder-decoder, T5 family) |
| Parameters | ~250M |
| Inference | Local, CPU, via `AutoTokenizer` + `AutoModelForSeq2SeqLM` (no `pipeline()` — removed `text2text-generation` support in transformers v5) |
| Max input length | 1024 tokens (prompt truncated if exceeded) |
| Max new tokens | 200 |
| Decoding | Greedy (`do_sample=False` behavior, default `.generate()` settings) |
| Cost | $0 — fully local, no API key |

## 7. Experiment Log (Requirement 8 — optimization experiments)

| Experiment | What changed | Observation |
|---|---|---|
| Baseline | chunk_size=500, overlap=50, vector-only search, top_k=3 | Retrieves relevant chunks reliably for direct, single-fact questions |
| Chunk size sweep | Tested chunk_size=300 vs 500 vs 800 (see README "Suggested experiments") | Smaller chunks (300) increase retrieval precision on narrow factual questions but can lose surrounding sentence context; larger chunks (800) give the LLM more context per chunk but dilute retrieval specificity |
| Hybrid search | Added optional BM25 + vector combination (`use_hybrid=True`) | Improves retrieval for queries containing exact keywords/names that the embedding model may under-weight semantically |
| Re-ranking | Added optional cross-encoder re-ranking over top-10 vector candidates, down-selecting to top-3 | Cross-encoders directly score (query, chunk) pairs rather than comparing independent embeddings, generally improving precision at the cost of extra inference time per query |

> To reproduce: toggle `USE_HYBRID` / `RERANK_MODEL` at the top of `rag_pipeline.py`'s
> `__main__` block, or pass `use_hybrid=True, rerank_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"`
> to `RAGPipeline(...)`, then re-run `validate.py` and compare `validation_log.md` output.

## 8. End-to-End Pipeline Summary

```
Document (.txt/.pdf/HF dataset)
   → normalize + chunk (500 chars, 50 overlap)
   → embed chunks (all-MiniLM-L6-v2, 384-dim)
   → store in FAISS (IndexFlatIP)
   → [user question]
   → embed question (same model)
   → retrieve top-k (vector, or hybrid + re-rank)
   → build prompt (context + question)
   → generate answer (flan-t5-base, local, CPU)
   → print grounded answer + retrieved sources
```
