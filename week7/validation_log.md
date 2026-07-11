# Validation Log
Generated: 2026-07-09T19:15:21
Document tested: `sample_document.txt`
Chunk size: 500, Overlap: 50, Top-k: 3

---

## Test 1
**Question:** What is Retrieval-Augmented Generation?

**Generated Answer:**
> combines information retrieval with text generation

**Retrieved Context (3 chunks):**
- `[1]` Introduction to Retrieval-Augmented Generation (RAG) Retrieval-Augmented Generation, or RAG, is a technique that combines information retrieval with text generation. Instead of relying purely on the i...
- `[2]` s heavily on the quality of retrieval. If the wrong chunks are retrieved, the generated answer may be irrelevant or incorrect. Chunking strategy, embedding model choice, and the number of retrieved ch...
- `[3]` ems that can answer questions using custom or private data. By combining a retrieval step with a generation step, RAG systems produce answers that are both relevant and grounded in real source materia...

---

## Test 2
**Question:** Why does RAG matter?

**Generated Answer:**
> It reduces hallucination

**Retrieved Context (3 chunks):**
- `[1]` ation, and academic research assistants that answer questions from papers. Advantages of RAG RAG reduces hallucination because answers are grounded in real retrieved text. It allows a model to answer ...
- `[2]` tain point in time and do not know about private, recent, or domain-specific information unless it is provided to them directly. RAG solves this problem by letting the model "look up" relevant facts b...
- `[3]` or this purpose and works well for small to medium-sized projects. Language models generate the final answer. For local, free experimentation, smaller models such as Google's FLAN-T5 family work well ...

---

## Test 3
**Question:** What embedding model is mentioned in the document?

**Generated Answer:**
> all-MiniLM-L6-v2

**Retrieved Context (3 chunks):**
- `[1]` , so the model can generate an answer that is grounded in the retrieved text. Common Components Embedding models convert text into vectors that capture semantic meaning. Popular choices include models...
- `[2]` s, and each chunk is converted into a numerical vector called an embedding. These embeddings are stored in a vector database that allows fast similarity search. In the querying phase, the user's quest...
- `[3]` Introduction to Retrieval-Augmented Generation (RAG) Retrieval-Augmented Generation, or RAG, is a technique that combines information retrieval with text generation. Instead of relying purely on the i...

---

## Test 4
**Question:** What vector database is mentioned and who developed it?

**Generated Answer:**
> FAISS

**Retrieved Context (3 chunks):**
- `[1]` , so the model can generate an answer that is grounded in the retrieved text. Common Components Embedding models convert text into vectors that capture semantic meaning. Popular choices include models...
- `[2]` s, and each chunk is converted into a numerical vector called an embedding. These embeddings are stored in a vector database that allows fast similarity search. In the querying phase, the user's quest...
- `[3]` ems that can answer questions using custom or private data. By combining a retrieval step with a generation step, RAG systems produce answers that are both relevant and grounded in real source materia...

---

## Test 5
**Question:** What are the limitations of RAG?

**Generated Answer:**
> The quality of answers depends heavily on the quality of retrieval.

**Retrieved Context (3 chunks):**
- `[1]` ation, and academic research assistants that answer questions from papers. Advantages of RAG RAG reduces hallucination because answers are grounded in real retrieved text. It allows a model to answer ...
- `[2]` or this purpose and works well for small to medium-sized projects. Language models generate the final answer. For local, free experimentation, smaller models such as Google's FLAN-T5 family work well ...
- `[3]` tain point in time and do not know about private, recent, or domain-specific information unless it is provided to them directly. RAG solves this problem by letting the model "look up" relevant facts b...

---

## Test 6
**Question:** What is a completely unrelated question with no answer in the document, like what is the capital of France?

**Generated Answer:**
> no

**Retrieved Context (3 chunks):**
- `[1]` s, and each chunk is converted into a numerical vector called an embedding. These embeddings are stored in a vector database that allows fast similarity search. In the querying phase, the user's quest...
- `[2]` ems that can answer questions using custom or private data. By combining a retrieval step with a generation step, RAG systems produce answers that are both relevant and grounded in real source materia...
- `[3]` s heavily on the quality of retrieval. If the wrong chunks are retrieved, the generated answer may be irrelevant or incorrect. Chunking strategy, embedding model choice, and the number of retrieved ch...

---
