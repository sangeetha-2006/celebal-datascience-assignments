"""
Step 5d: LangGraph agent.

Orchestrates the full pipeline for one user turn:

    load_memory -> retrieve_rag -> query_kg -> [maybe_call_tool] -> generate_answer -> save_memory

The routing decision (whether a live tool is needed) is made by a simple
keyword heuristic in app/agent/tools.py — this is the "dynamic intelligence
layer" described in the problem statement, deciding between static knowledge
(RAG/KG) and dynamic tools.
"""
from langgraph.graph import END, StateGraph

from app.agent.llm import generate
from app.agent.preferences import extract_and_save_preferences
from app.agent.state import AgentState
from app.agent.tools import select_tool
from app.graph.query_graph import query_entity
from app.memory.memory_store import add_message, get_memory_context
from app.rag.retriever import retrieve

MAX_HISTORY_FOR_PROMPT = 12


# ---- Nodes ----

def load_memory_node(state: AgentState) -> AgentState:
    # Extract & save any explicit preference statements first (e.g. "my name
    # is X") so this turn's memory_context already reflects the update.
    extract_and_save_preferences(state["user_id"], state["query"])
    context = get_memory_context(state["user_id"], history_limit=MAX_HISTORY_FOR_PROMPT)
    return {**state, "memory_context": context}


def retrieve_rag_node(state: AgentState) -> AgentState:
    try:
        results = retrieve(state["query"], top_k=4)
        formatted = "\n".join(f"- {r['text'][:300]}" for r in results) or "No relevant documents found."
    except FileNotFoundError:
        formatted = "RAG index not built yet (run: python -m app.rag.build_index)."
    return {**state, "rag_context": formatted}


def query_kg_node(state: AgentState) -> AgentState:
    try:
        results = query_entity(state["query"])
        if results:
            formatted = "\n".join(
                f"- ({r['source']}) -[{r['relation']}]-> ({r['target']})" for r in results[:5]
            )
        else:
            formatted = "No relevant graph relationships found."
    except Exception as e:  # noqa: BLE001 - surface any Neo4j connectivity issue as context, not a crash
        formatted = f"Knowledge graph unavailable ({e})."
    return {**state, "kg_context": formatted}


def router(state: AgentState) -> str:
    """Decide whether this query needs a live tool."""
    return "call_tool" if select_tool(state["query"], state["user_id"]) else "skip_tool"


def call_tool_node(state: AgentState) -> AgentState:
    selection = select_tool(state["query"], state["user_id"])
    if selection is None:
        return {**state, "tool_context": ""}
    tool_name, tool_fn = selection
    result = tool_fn(state["query"])
    return {**state, "tool_context": f"[{tool_name} tool result]\n{result}"}


def skip_tool_node(state: AgentState) -> AgentState:
    return {**state, "tool_context": "No live tool was needed for this query."}


def generate_answer_node(state: AgentState) -> AgentState:
    has_tool_result = state["tool_context"] and not state["tool_context"].startswith("No live tool")
    has_rag = state["rag_context"] and not state["rag_context"].startswith(
        ("RAG index not built", "No relevant")
    )
    has_kg = state["kg_context"] and not state["kg_context"].startswith(
        ("No relevant", "Knowledge graph unavailable")
    )

    sections = []

    if has_tool_result:
        sections.append(f"LIVE DATA (use this to answer):\n{state['tool_context']}")
    if has_rag:
        sections.append(f"RELEVANT DOCUMENTS:\n{state['rag_context']}")
    if has_kg:
        sections.append(f"KNOWN FACTS (graph):\n{state['kg_context']}")

    # Conversation history is labeled explicitly and put right next to the
    # question, since small models attend far more reliably to text near
    # the end of the prompt than to text buried earlier.
    history_block = state.get("memory_context", "")

    context_block = "\n\n".join(sections) if sections else "(no retrieved context for this question)"

    prompt = f"""Follow these rules exactly:
1. Answer ONLY what is asked. Do not add unrelated facts or extra trivia.
2. You are the assistant. The person asking is "the user". Address them as \
"you"/"your" — never say "I" or "my" when referring to facts about the user \
(e.g. if the user's name is Sam, say "Your name is Sam," not "My name is Sam").
3. If USER PREFERENCES below lists a fact (like name or favorite language), \
treat it as true and current — it is always more reliable than anything in \
CONVERSATION HISTORY.
4. If asked whether something was discussed earlier, look at CONVERSATION HISTORY \
below and answer based on what is actually there — do not guess.
5. If CONTEXT below mentions or defines the specific thing being asked about, you MUST \
base your answer on that — even if you think you already know the term from elsewhere. \
The definition in CONTEXT always overrides your own prior assumption about a term, \
because CONTEXT describes THIS specific project, not the general/generic meaning of the word.
6. If CONTEXT is not relevant to the question at all, ignore it and answer from general knowledge.
7. If you don't know the answer, say so plainly. Never invent facts.
8. Keep the answer short — 1 to 3 sentences unless more detail is clearly needed.

CONTEXT:
{context_block}

{history_block}

QUESTION: {state['query']}

ANSWER:"""
    answer = generate(prompt)
    return {**state, "answer": answer}


def save_memory_node(state: AgentState) -> AgentState:
    add_message(state["user_id"], "user", state["query"])
    add_message(state["user_id"], "assistant", state["answer"])
    return state


# ---- Graph construction ----

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("retrieve_rag", retrieve_rag_node)
    graph.add_node("query_kg", query_kg_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("skip_tool", skip_tool_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "retrieve_rag")
    graph.add_edge("retrieve_rag", "query_kg")
    graph.add_conditional_edges(
        "query_kg",
        router,
        {"call_tool": "call_tool", "skip_tool": "skip_tool"},
    )
    graph.add_edge("call_tool", "generate_answer")
    graph.add_edge("skip_tool", "generate_answer")
    graph.add_edge("generate_answer", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


def run_agent(user_id: str, query: str) -> str:
    """Convenience function: run one turn through the compiled agent."""
    app = build_agent()
    result = app.invoke({"user_id": user_id, "query": query})
    return result["answer"]


def run_agent_full(user_id: str, query: str) -> AgentState:
    """Like run_agent, but returns the full final state (including
    rag_context, kg_context, tool_context) — used by the evaluation
    framework (Phase 6) to check what context the answer was based on.
    """
    app = build_agent()
    return app.invoke({"user_id": user_id, "query": query})
