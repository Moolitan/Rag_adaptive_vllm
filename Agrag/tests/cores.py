# -*- coding: utf-8 -*-
"""
Shared helpers for benchmark runners.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from runner.engine import build_app, run_one, summarize
from runner.logging import log, C, safe_preview, now_ts


# -------------------------
# Models
# -------------------------
@dataclass
class Example:
    id: str
    question: str
    answer: str
    meta: Dict[str, Any]
    custom_retriever_config: Optional[Dict[str, str]] = None


# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: str) -> List[dict]:
    """
    Read JSONL files that may contain either one-JSON-per-line (compact) or pretty
    multi-line JSON objects separated by blank lines.
    """
    rows: List[dict] = []
    buf: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                if buf:
                    rows.append(json.loads("".join(buf)))
                    buf = []
                continue
            if not buf:
                try:
                    rows.append(json.loads(ln))
                    continue
                except json.JSONDecodeError:
                    buf.append(ln)
            else:
                buf.append(ln)
    if buf:
        rows.append(json.loads("".join(buf)))
    return rows


def write_jsonl(path: str, rows: List[dict], pretty: bool = False) -> None:
    """
    Write JSONL in either compact (default) or pretty multi-line mode.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            if pretty:
                json.dump(r, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get(d: dict, *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# -------------------------
# Answer matching
# -------------------------
def norm_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`.,;:!?()\\[\\]{}]", "", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> bool:
    return norm_answer(pred) == norm_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    p = norm_answer(pred)
    g = norm_answer(gold)
    return (g in p) if (g and p) else False


# -------------------------
# Failure buckets
# -------------------------
def bucket_failure(row: dict) -> str:
    if row.get("correct"):
        return "OK"

    grounded = row.get("grounded", "no")
    useful = row.get("useful", "no")
    graded_docs = int(row.get("graded_relevant_docs", 0) or 0)
    has_irrelevant = bool(row.get("has_irrelevant_docs", False))

    if grounded != "yes":
        return "FAIL_grounding"
    if has_irrelevant:
        return "FAIL_doc_quality"
    if graded_docs == 0:
        return "FAIL_retrieval"
    if useful != "yes":
        return "FAIL_useful"
    return "FAIL_answer"


def bucket_stats(rows: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        b = r.get("bucket", "UNKNOWN")
        out[b] = out.get(b, 0) + 1
    return out

def load_hotpotqa_fullwiki(
    path: str,
    limit: int | None,
    persist_dir: str,
    collection_name: str,
    retrieval_k: int,
) -> list[Example]:
    """
    Load HotpotQA fullwiki data without using provided context.
    Instead, configure custom retriever for Chroma-based retrieval.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exs: list[Example] = []
    custom_retriever_config = {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "k": str(retrieval_k),
    }

    for i, row in enumerate(data):
        qid = str(row.get("_id", i))
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()

        # Do NOT use provided context - let the system retrieve from Chroma
        exs.append(Example(id=qid, question=q, answer=a, meta=row, custom_retriever_config=custom_retriever_config))
        if limit and len(exs) >= limit:
            break

    return exs


# -------------------------
# Core runner
# -------------------------
def run_benchmark(
    task: str,
    examples: List[Example],
    out_path: str,
    soft_match: bool = False,
    rag_name: str = "agrag",
    handle: Any | None = None,
) -> Dict[str, Any]:
    # Print runtime configuration once at the start
    log("CONFIG", "=" * 60, C.YELLOW)
    log("CONFIG", f"Task: {task}", C.YELLOW)
    log("CONFIG", f"Examples: {len(examples)}", C.YELLOW)
    log("CONFIG", f"Output: {out_path}", C.YELLOW)
    log("CONFIG", f"RAG: {rag_name}", C.YELLOW)
    log("CONFIG", "=" * 60, C.YELLOW)

    handle = handle or build_app(rag_name=rag_name)
    results: List[dict] = []
    t0 = time.time()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Truncate/prepare output file; incremental writes happen per example.
    out_file.write_text("", encoding="utf-8")
    pretty_path = out_file.with_suffix(".pretty.json")

    try:
        for idx, ex in enumerate(examples):
            thread_id = f"bench_{task}_{ex.id}_{idx}"
            print("\n")
            log("Benchmark", f"Question: {ex.question}", C.RED)
            out = run_one(
                handle,
                question=ex.question,
                thread_id=thread_id,
                custom_retriever_config=ex.custom_retriever_config,
            )

            pred = out["prediction"]
            gold = ex.answer
            correct = exact_match(pred, gold) or (soft_match and contains_match(pred, gold))

            # Show predicted and gold answers during the run for quick inspection.
            log("Benchmark", f"Prediction: {safe_preview(str(pred), 200)}", C.GREEN)
            log("Benchmark", f"Gold: {safe_preview(str(gold), 200)}", C.CYAN)
            log("Benchmark", f"Correct: {'yes' if correct else 'no'}", C.YELLOW)

            row = {
                "task": task,
                "id": ex.id,
                "question": ex.question,
                "gold": gold,
                "prediction": pred,
                "supporting_facts": out.get("final_state", {}).get("supporting_facts", []),
                "meta": ex.meta,
                "correct": correct,
                "router_decision": out.get("router_decision", ""),
                "data_source": out.get("data_source", ""),
                "graded_relevant_docs": out.get("graded_relevant_docs", 0),
                "grounded": out.get("grounded", "no"),
                "useful": out.get("useful", "no"),
                "used_web_search": out.get("used_web_search", False),
                "web_search_rounds": out.get("web_search_rounds", 0),
                "has_irrelevant_docs": out.get("has_irrelevant_docs", False),
            }
            row["bucket"] = bucket_failure(row)
            results.append(row)
            with open(out_file, "a", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
    finally:
        try:
            handle.close()
        except Exception:
            pass

    dt = time.time() - t0
    stats = summarize(results)
    buckets = bucket_stats(results)
    # Re-write full file on success to ensure consistency; partial results stay if interrupted.
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
    # Also emit a human-readable pretty JSON bundle alongside the line-delimited file.
    with open(pretty_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "stats": stats, "buckets": buckets, "elapsed_sec": dt},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"results": results, "stats": stats, "buckets": buckets, "elapsed_sec": dt}
