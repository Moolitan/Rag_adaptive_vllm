# -*- coding: utf-8 -*-
"""
Reusable runner utilities for Agentic RAG.

Design goals:
- No CLI parsing (argparse) inside.
- No interactive I/O (input/print banner) inside.
- Provide pure functions for scripts/tests to call.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Rag import (
    get_ag_rag_app,
    get_c_rag_app,
    get_hop2_rag_app,
    get_self_rag_app,
)
from runner.logging import C


# RAG名称映射到构建函数
_RAG_BUILDERS = {
    "agrag": get_ag_rag_app,
    "ag_rag": get_ag_rag_app,
    "crag": get_c_rag_app,
    "c_rag": get_c_rag_app,
    "hop2rag": get_hop2_rag_app,
    "hop2_rag": get_hop2_rag_app,
    "selfrag": get_self_rag_app,
    "self_rag": get_self_rag_app,
}


def list_rags() -> List[str]:
    """列出所有可用的RAG类型"""
    return list(_RAG_BUILDERS.keys())

@dataclass
class AppHandle:
    """Hold compiled app."""
    app: Any

    def close(self):
        pass


def build_app(rag_name: str = "agrag") -> AppHandle:
    """Build app from a registered RAG recipe (default: agrag)."""
    rag_name_lower = rag_name.lower()
    if rag_name_lower not in _RAG_BUILDERS:
        available = list_rags()
        raise ValueError(f"Unknown RAG '{rag_name}'. Available: {available}")

    builder = _RAG_BUILDERS[rag_name_lower]
    app = builder()
    return AppHandle(app=app)


def run_one(
    handle: AppHandle,
    question: str,
    thread_id: str,
    custom_retriever_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    inputs: Dict[str, Any] = {
        "question": question,
        "loop": 0,
        "web_search_rounds": 0,
    }
    if custom_retriever_config:
        inputs["custom_retriever_config"] = custom_retriever_config

    final_state = handle.app.invoke(inputs, config=config)

    pred = (final_state.get("generation") or "").strip()
    return {
        "final_state": final_state,
        "prediction": pred,
        "router_decision": final_state.get("router_decision", ""),
        "data_source": final_state.get("data_source", ""),
        "graded_relevant_docs": int(final_state.get("graded_relevant_docs", 0) or 0),
        "grounded": final_state.get("grounded", "no"),
        "useful": final_state.get("useful", "no"),
        "used_web_search": bool(final_state.get("used_web_search", False)),
        "web_search_rounds": int(final_state.get("web_search_rounds", 0) or 0),
        "has_irrelevant_docs": bool(final_state.get("has_irrelevant_docs", False)),
    }


def save_workflow_png(app_or_handle, output_file: str = "agent_workflow.png") -> str:
    """
    Save workflow graph visualization as PNG. Returns absolute path.
    If PNG generation fails, saves mermaid syntax as .mmd and returns its path.

    Accepts either:
      - compiled app (has .get_graph())
      - AppHandle (has .app)
    """
    app = getattr(app_or_handle, "app", app_or_handle)

    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open(output_file, "wb") as f:
            f.write(png_data)
        return os.path.abspath(output_file)
    except Exception:
        mmd_file = os.path.splitext(output_file)[0] + ".mmd"
        with open(mmd_file, "w", encoding="utf-8") as f:
            f.write(app.get_graph().draw_mermaid())
        return os.path.abspath(mmd_file)



def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics for benchmark runs."""
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "accuracy": sum(1 for r in rows if r.get("correct")) / n,
        "grounded_yes_rate": sum(1 for r in rows if r.get("grounded") == "yes") / n,
        "useful_yes_rate": sum(1 for r in rows if r.get("useful") == "yes") / n,
        "used_web_rate": sum(1 for r in rows if r.get("used_web_search")) / n,
        "no_relevant_docs_rate": sum(1 for r in rows if int(r.get("graded_relevant_docs", 0) or 0) == 0) / n,
        "rerouted_for_quality_rate": sum(1 for r in rows if r.get("has_irrelevant_docs")) / n,
    }
