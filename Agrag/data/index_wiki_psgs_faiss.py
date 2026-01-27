#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS vectorstore from DPR-style Wikipedia passages TSV.

Expected TSV header: id, text, title

Usage:
  python tests/wiki/index_wiki_psgs_faiss.py \
    --tsv-path tests/data/wiki/psgs_w100.tsv \
    --out-dir /mnt/Large_Language_Model_Lab_1/faiss_wiki_db \
    --batch-size 5120 \
    --embed-batch-size 1024 \
    --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import json
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def iter_docs_from_tsv(path: Path, max_docs: int) -> Iterable[Document]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, 1):
            if max_docs > 0 and i > max_docs:
                break
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if not text:
                continue
            content = f"{title}\n{text}" if title else text
            yield Document(
                page_content=content,
                metadata={
                    "source": "wiki_psgs_w100",
                    "id": str(row.get("id") or i),
                    "title": title,
                },
            )


def batched(iterable: Iterable[Document], batch_size: int) -> Iterable[List[Document]]:
    batch: List[Document] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def split_tsv_into_chunks(
    tsv_path: Path,
    chunk_dir: Path,
    chunk_size: int,
    max_docs: int,
) -> List[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunk_dir.glob("chunk_*.tsv"))
    if existing:
        return existing

    chunks: List[Path] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or ["id", "text", "title"]
        chunk_idx = 0
        rows_in_chunk = 0
        out_f = None
        writer = None
        total = 0

        for row in reader:
            if max_docs > 0 and total >= max_docs:
                break
            if rows_in_chunk == 0:
                chunk_idx += 1
                chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.tsv"
                out_f = chunk_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(out_f, fieldnames=header, delimiter="\t")
                writer.writeheader()
                chunks.append(chunk_path)
            writer.writerow(row)
            rows_in_chunk += 1
            total += 1
            if rows_in_chunk >= chunk_size:
                out_f.close()
                out_f = None
                writer = None
                rows_in_chunk = 0

        if out_f is not None:
            out_f.close()

    return chunks


def load_progress(progress_path: Path) -> int:
    if not progress_path.is_file():
        return 0
    with progress_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("last_chunk", 0))


def save_progress(progress_path: Path, last_chunk: int) -> None:
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump({"last_chunk": last_chunk}, f)


def build_faiss(
    tsv_path: Path,
    out_dir: Path,
    batch_size: int,
    embed_batch_size: int,
    model_name: str,
    device: str,
    max_docs: int,
    chunk_size: int,
) -> None:
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": embed_batch_size,
        },
    )

    os.makedirs(out_dir, exist_ok=True)
    progress_path = out_dir / "progress.json"
    chunk_dir = out_dir / "chunks"

    vectorstore: Optional[FAISS] = None
    if (out_dir / "index.faiss").is_file():
        vectorstore = FAISS.load_local(
            str(out_dir),
            embedding,
            allow_dangerous_deserialization=True,
        )

    chunks = split_tsv_into_chunks(tsv_path, chunk_dir, chunk_size, max_docs)
    last_chunk = load_progress(progress_path)

    total_docs = 0
    for idx, chunk_path in enumerate(chunks, 1):
        if idx <= last_chunk:
            continue
        docs_iter = iter_docs_from_tsv(chunk_path, max_docs=0)
        for batch in batched(docs_iter, batch_size):
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embedding)
            else:
                vectorstore.add_documents(batch)
            total_docs += len(batch)
            if total_docs % (batch_size * 10) == 0:
                print(f"[PROGRESS] indexed {total_docs} docs")

        if vectorstore is None:
            raise RuntimeError("No documents were indexed.")

        vectorstore.save_local(str(out_dir))
        save_progress(progress_path, idx)
        print(f"[OK] saved chunk {idx}/{len(chunks)} to {out_dir}")

    if vectorstore is None:
        raise RuntimeError("No documents were indexed.")

    print(f"[OK] saved FAISS index to {out_dir} with {total_docs} docs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS index for wiki passages TSV")
    ap.add_argument(
        "--tsv-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests" / "data" / "wiki" / "psgs_w100.tsv",
        help="Path to psgs_w100.tsv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/Large_Language_Model_Lab_1/faiss_wiki_db"),
        help="FAISS output directory",
    )
    ap.add_argument("--batch-size", type=int, default=5120)
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=1024,
        help="Embedding batch size (controls GPU memory usage).",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--model-name",
        type=str,
        default="/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5",
        help="Embedding model name or local path. If empty, uses $BGE_MODEL_PATH.",
    )
    ap.add_argument("--max-docs", type=int, default=0, help="Limit total documents")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Number of rows per TSV chunk for incremental indexing.",
    )
    args = ap.parse_args()

    model_name = args.model_name or os.environ.get("BGE_MODEL_PATH", "")
    if not model_name:
        raise ValueError(
            "Missing embedding model. Set --model-name or export BGE_MODEL_PATH."
        )

    build_faiss(
        args.tsv_path,
        args.out_dir,
        args.batch_size,
        args.embed_batch_size,
        model_name,
        args.device,
        args.max_docs,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
