"""
Step 5a: Agent state.

Defines the shared state that flows through every node in the LangGraph
workflow. Each node reads from and adds to this state.
"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    query: str

    memory_context: str      # from app.memory (Phase 4)
    rag_context: str         # from app.rag (Phase 2)
    kg_context: str          # from app.graph (Phase 3)
    tool_context: str        # from a live tool (Phase 5, dynamic)

    needs_tool: bool         # decided by the router

    answer: str              # final generated response
