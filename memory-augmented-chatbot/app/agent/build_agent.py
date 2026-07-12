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
from app.agent.state import AgentState
from app.agent.tools import select_tool
from app.graph.query_graph import query_entity
from app.memory.memory_store import add_message, get_memory_context
from app.rag.retriever import retrieve

MAX_HISTORY_FOR_PROMPT = 6


# ---- Nodes ----

def load_memory_node(state: AgentState) -> AgentState:
    context = get_memory_context(state["user_id"], history_limit=MAX_HISTORY_FOR_PROMPT)
    return {**state, "memory_context": context}


def retrieve_rag_node(state: AgentState) -> AgentState:
    try:
        results = retrieve(state["query"], top_k=2)
        formatted = "\n".join(f"- {r['text'][:150]}" for r in results) or "No relevant documents found."
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
    return "call_tool" if select_tool(state["query"]) else "skip_tool"


def call_tool_node(state: AgentState) -> AgentState:
    selection = select_tool(state["query"])
    if selection is None:
        return {**state, "tool_context": ""}
    tool_name, tool_fn = selection
    result = tool_fn(state["query"])
    return {**state, "tool_context": f"[{tool_name} tool result]\n{result}"}


def skip_tool_node(state: AgentState) -> AgentState:
    return {**state, "tool_context": "No live tool was needed for this query."}


def generate_answer_node(state: AgentState) -> AgentState:
    has_tool_result = state["tool_context"] and not state["tool_context"].startswith("No live tool")

    sections = []

    if has_tool_result:
        # Put the live tool result first and make it unmissable — small models
        # tend to ignore context that's buried after a lot of other text.
        sections.append(
            f"IMPORTANT — use this real-time information to answer the question:\n{state['tool_context']}"
        )

    sections.append(f"Background knowledge:\n{state['rag_context']}")
    sections.append(f"Known relationships:\n{state['kg_context']}")

    if state["memory_context"] and "none recorded" not in state["memory_context"]:
        sections.append(state["memory_context"])

    context_block = "\n\n".join(sections)

    prompt = f"""Answer the user's question directly and concisely, using the context below \
if it's relevant. If the context doesn't help, answer from your own knowledge. \
Do not mention that you are using "context" or "background knowledge" — just answer naturally.

{context_block}

Question: {state['query']}

Answer:"""
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
