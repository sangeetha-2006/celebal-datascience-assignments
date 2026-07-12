# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

End-to-end system combining RAG, a knowledge graph, long-term user memory, and
LangGraph-based tool routing for context-aware, personalized responses.

This repo is being built in phases. **Phase 1 (this drop): project scaffold + data ingestion pipeline.**

## Project Structure

```
memchat/
├── app/
│   ├── config.py            # centralized settings (reads .env)
│   ├── ingestion/           # Phase 1: scrape -> clean -> chunk
│   │   ├── scraper.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── run_pipeline.py
│   ├── rag/                 # Phase 2: embeddings + vector search (next)
│   ├── graph/                # Phase 3: Neo4j knowledge graph (next)
│   ├── memory/               # Phase 4: long-term user memory (next)
│   ├── agent/                 # Phase 5: LangGraph orchestration (next)
│   ├── eval/                  # Phase 6: evaluation framework (next)
│   └── api/                   # Phase 7: FastAPI endpoints (next)
├── data/
│   ├── raw/                 # raw scraped JSON, one file per page
│   ├── processed/           # cleaned.jsonl, chunks.jsonl
│   └── vectorstore/         # FAISS/Chroma index files (generated)
├── tests/
├── requirements.txt
├── .env.example
└── urls_example.txt
```

## Setup

1. **Install Python 3.10+**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Copy the env template** and fill in your keys as you get them:
   ```bash
   cp .env.example .env
   ```
   You don't need any keys yet for Phase 1 (scraping/cleaning/chunking runs with
   no API keys). You'll add `OPENAI_API_KEY` in Phase 2 and Neo4j credentials in
   Phase 3.

## Running Phase 1 (Data Pipeline)

Put one URL per line in a text file (see `urls_example.txt` for a sample), then:

```bash
python -m app.ingestion.run_pipeline --urls urls_example.txt
```

This runs all three steps in order and produces:
- `data/raw/*.json` — one raw scrape per URL
- `data/processed/cleaned.jsonl` — cleaned text, one doc per line
- `data/processed/chunks.jsonl` — chunked text ready for embedding

You can also run each step individually:
```bash
python -m app.ingestion.scraper --urls urls_example.txt
python -m app.ingestion.cleaner
python -m app.ingestion.chunker
```

## Running Phase 2 (RAG: Embeddings + Vector Search)

Phase 2 uses a **local, free embedding model** (`all-MiniLM-L6-v2` via
`sentence-transformers`) — no API key needed. On the very first run it
downloads the model (~90MB) from Hugging Face and caches it in
`~/.cache/huggingface`; after that it works fully offline.

1. Make sure you've already run Phase 1 so `data/processed/chunks.jsonl` exists.

2. Build the vector index:
   ```bash
   python -m app.rag.build_index
   ```
   This embeds every chunk and saves the index to `data/vectorstore/`
   (`index.faiss` + `metadata.jsonl`).

3. Query it:
   ```bash
   python -m app.rag.retriever "What is retrieval-augmented generation?"
   ```
   This prints the top 5 most relevant chunks with similarity scores.

Files:
- `app/rag/embedder.py` — loads the local model, embeds text
- `app/rag/vectorstore.py` — FAISS wrapper (add / search / save / load)
- `app/rag/build_index.py` — builds the index from `chunks.jsonl`
- `app/rag/retriever.py` — embeds a query and searches the index

## Running Phase 3 (Knowledge Graph: Entity Extraction + Neo4j)

Phase 3 uses **spaCy** (free, offline, no API key) to extract (entity, relation,
entity) triples from your scraped documents via dependency parsing, then loads
them into Neo4j as a graph.

This is a lighter-weight heuristic approach than using an LLM — it won't catch
every relationship an LLM would, but it's completely free and runs locally.

**Requires:** `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` set in your
`.env`. No OpenAI key needed for this phase.

1. Install spaCy and its small English model (one-time):
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

2. Make sure Phase 1 has already run so `data/processed/cleaned.jsonl` exists.

3. Build the graph:
   ```bash
   python -m app.graph.build_graph
   ```
   Add `--limit 1` to test on just one document first, or `--clear-first` to
   wipe the graph before rebuilding.

4. Query it:
   ```bash
   python -m app.graph.query_graph "RAG"
   ```
   This prints every relationship in the graph involving anything matching
   "RAG" (case-insensitive substring match).

5. You can also explore visually: log into https://console.neo4j.io, open your
   instance, go to **Query**, and run:
   ```cypher
   MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100
   ```

Files:
- `app/graph/neo4j_client.py` — connection + write/query helpers
- `app/graph/extractor.py` — spaCy-based triple extraction (dependency parsing)
- `app/graph/build_graph.py` — runs extraction over all documents, loads into Neo4j
- `app/graph/query_graph.py` — CLI to query the graph

**Cost note:** entirely free — no API key or billing required for this phase.

## Running Phase 4 (Long-Term Memory)

Phase 4 uses **SQLite** — built into Python, zero setup, no account or server
needed. It stores durable user preferences and full conversation history in a
single file: `data/memory.db`.

(Note: the original problem statement suggested MongoDB/Postgres. SQLite was
chosen instead for simplicity — same purpose, no infrastructure to manage.
Swapping in MongoDB/Postgres later would only mean changing `app/memory/db.py`.)

1. Try the demo (creates the DB automatically on first run):
   ```bash
   python -m app.memory.demo
   ```
   This stores a couple of preferences and messages for a demo user, then
   prints the assembled "memory context" — the text block that Phase 5 (the
   LangGraph agent) will feed into the LLM so it can personalize responses.

2. Run it again — you'll see history accumulate across runs, since it's
   reading from the same `data/memory.db` file each time.

Files:
- `app/memory/db.py` — SQLite connection + schema (users, preferences, messages)
- `app/memory/memory_store.py` — read/write functions (`set_preference`,
  `get_preferences`, `add_message`, `get_recent_history`, `get_memory_context`)
- `app/memory/demo.py` — a runnable demo showing it all working together

**Cost note:** entirely free, no dependencies beyond Python's standard library.

## Running Phase 5 (LangGraph Agent: RAG + Knowledge Graph + Memory + Tools)

This is where everything comes together. The agent is a LangGraph workflow
that, for every user message:

1. **Loads memory** (Phase 4) — preferences + recent conversation history
2. **Retrieves from RAG** (Phase 2) — relevant chunks from the vector store
3. **Queries the knowledge graph** (Phase 3) — relevant entity relationships
4. **Routes to a live tool if needed** (e.g. weather, current time) — a simple
   keyword-based router decides this; a more advanced system could use an
   LLM for the routing decision itself
5. **Generates the final answer** using a local Ollama model (fully free, no
   API key, no quota — runs entirely on your machine), given all of the
   above as context
6. **Saves the exchange back to memory** for future turns

**Requires:** [Ollama](https://ollama.com/download) installed and running,
with a model pulled (`ollama pull llama3.2:1b`). RAG and Neo4j are used if
available, but the agent degrades gracefully and still answers if either
isn't set up yet (it'll just say so in its reasoning).

1. Install Ollama (https://ollama.com/download) and pull a model:
   ```bash
   ollama pull llama3.2:1b
   ```
   (Use `ollama pull llama3.2` instead for the larger, higher-quality 3B
   version if your machine has 16GB+ RAM.)

2. Chat with it interactively:
   ```bash
   python -m app.agent.chat
   ```
   Type a question, get an answer, type `exit` to quit. Try asking about
   something from your scraped documents (e.g. "What is RAG?"), then try
   something time-sensitive (e.g. "What time is it right now?") to see the
   tool-routing kick in.

Files:
- `app/agent/state.py` — the shared state passed between nodes
- `app/agent/tools.py` — free dynamic tools (current time, weather via wttr.in)
  and the keyword-based tool router
- `app/agent/llm.py` — Ollama wrapper for response generation (local HTTP call)
- `app/agent/build_agent.py` — the LangGraph workflow (nodes + routing)
- `app/agent/chat.py` — interactive CLI

**Cost note:** entirely free — Ollama runs the model locally, no account,
no API key, no usage limits.

## Running Phase 6 (Evaluation Framework)

Phase 6 measures response quality automatically — no manual grading, and no
paid API. It reuses tools you already have:

- **Context relevance** — cosine similarity between the question and the
  RAG chunks retrieved for it (uses the same free local embedding model
  from Phase 2). High = the retriever pulled back genuinely relevant material.
- **Answer relevance** — cosine similarity between the question and the
  generated answer. High = the answer is actually on-topic.
- **Faithfulness** — your local Ollama model (Phase 5) acts as a free
  "LLM judge," scoring 0–1 on whether the answer's claims are actually
  supported by the retrieved context, or made up.

This mirrors what a library like RAGAS measures, implemented locally so
everything stays free.

1. Make sure Phases 1, 2, and 5 are working (RAG index built, Ollama running).
   Neo4j is optional for this phase.

2. Run the evaluation:
   ```bash
   python -m app.eval.run_eval
   ```
   This runs every question in `app/eval/test_set.py` through the full
   agent, scores each response, and prints a summary. Full per-question
   results are saved to `data/eval_results.json`.

3. Edit `app/eval/test_set.py` to add your own questions relevant to
   whatever you've scraped.

Files:
- `app/eval/metrics.py` — context relevance, answer relevance, faithfulness
- `app/eval/test_set.py` — the list of questions to evaluate against
- `app/eval/run_eval.py` — runs the agent + scores + saves a report

**Cost note:** entirely free — reuses the local embedding model and your
local Ollama model, no paid API calls.

## Running Phase 7 (API Layer)

Phase 7 exposes the entire system as a web API using FastAPI, so it can be
called from a frontend, a mobile app, curl, Postman, or anything else that
can make an HTTP request.

1. Start the server:
   ```bash
   uvicorn app.api.main:app --reload
   ```

2. Open **http://127.0.0.1:8000/docs** in your browser — FastAPI
   auto-generates interactive API documentation where you can try every
   endpoint directly from the browser.

**Endpoints:**

| Method | Path              | Description                                        |
|--------|-------------------|-----------------------------------------------------|
| GET    | `/health`         | Liveness check                                      |
| POST   | `/chat`           | Send `{user_id, query}`, get `{answer}` — runs the full agent pipeline |
| GET    | `/memory/{user_id}` | See stored preferences + recent history for a user |
| GET    | `/rag/search?q=...` | Directly search the RAG vector store               |
| GET    | `/kg/search?entity=...` | Directly query the knowledge graph              |

Example with curl:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "curl_user", "query": "What is RAG?"}'
```

Files:
- `app/api/main.py` — the FastAPI app and all routes
- `app/api/schemas.py` — Pydantic request/response models

**Cost note:** entirely free — FastAPI/uvicorn are just a web server running
locally, no external service involved.

---

## Project complete

All 7 phases are now implemented, end-to-end, entirely free:

1. Data ingestion pipeline (scrape → clean → chunk)
2. RAG (local embeddings + FAISS vector search)
3. Knowledge graph (spaCy extraction + Neo4j)
4. Long-term memory (SQLite)
5. LangGraph agent (routes between RAG, KG, memory, live tools; generates
   with local Ollama)
6. Evaluation framework (context relevance, answer relevance, faithfulness)
7. FastAPI web API

No paid API keys or billing were required anywhere in this build.
