#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HotpotQA Full Wiki Indexing Script

Reads Wikipedia intro paragraphs from enwiki-20171001 bz2 files,
chunks them with sentence numbering (for supporting facts alignment),
and indexes into Chroma for retrieval.

Usage:
    python tests/HotpotQA/index_hotpotqa_fullwiki.py \
        --wiki-dir tests/HotpotQA/data/enwiki-20171001-pages-meta-current-withlinks-abstracts \
        --persist-dir ./chroma_db_hotpotqa_fullwiki \
        --collection hotpotqa_fullwiki \
        --limit 0 \
        --batch-size 500
"""
import argparse
import bz2
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]  # Agrag/tests/data/hotpotqa/xxx.py -> Agrag/
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
# Option 1: Nomic (current, local, 768-dim)
from langchain_nomic.embeddings import NomicEmbeddings
# Option 2: BGE (stronger, local, 1024-dim) - uncomment to use
# from langchain_huggingface import HuggingFaceEmbeddings
# Option 3: OpenAI (strongest, API, 3072-dim) - uncomment to use
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    Works reasonably well for Wikipedia intro paragraphs.
    """
    # Simple sentence splitting: split on '. ' but avoid common abbreviations
    # This is a simplified version - for production, consider using spaCy or NLTK
    text = text.strip()
    if not text:
        return []

    # Split on period followed by space and capital letter, or period at end
    # Also handle question marks and exclamation marks
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Clean up and filter empty sentences
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) > 3:  # Ignore very short fragments
            cleaned.append(s)

    return cleaned if cleaned else [text]


def process_wiki_article(article: Dict[str, Any]) -> List[Document]:
    """
    Convert a single Wikipedia article into Document objects.
    Each paragraph becomes a separate document with numbered sentences.

    Format matches HotpotQA context format:
        Title
        0. First sentence.
        1. Second sentence.
        ...
    """
    docs = []

    wiki_id = str(article.get("id", ""))
    title = str(article.get("title", "")).strip()
    text_array = article.get("text", [])

    if not title or not text_array:
        return docs

    # Process each paragraph separately
    for para_idx, paragraph in enumerate(text_array):
        if not paragraph or not isinstance(paragraph, str):
            continue

        paragraph = paragraph.strip()
        if len(paragraph) < 20:  # Skip very short paragraphs
            continue

        # Split into sentences and number them
        sentences = sentence_split(paragraph)
        if not sentences:
            continue

        # Format: numbered sentences
        numbered_lines = []
        for sent_idx, sent in enumerate(sentences):
            numbered_lines.append(f"{sent_idx}. {sent}")

        # Combine: Title\n0. Sent1\n1. Sent2\n...
        content_lines = [title] + numbered_lines
        page_content = "\n".join(content_lines)

        # Create document with metadata
        doc = Document(
            page_content=page_content,
            metadata={
                "source": "hotpotqa_fullwiki",
                "title": title,
                "doc_id": wiki_id,
                "para_idx": para_idx,
                "num_sentences": len(sentences),
            }
        )
        docs.append(doc)

    return docs


def load_wiki_articles(wiki_dir: Path, limit: int = 0) -> List[Document]:
    """
    Load all Wikipedia articles from bz2 files in wiki_dir.
    Each bz2 file contains JSONL (one JSON object per line).

    Args:
        wiki_dir: Directory containing AA/, AB/, ..., EP/ subdirectories with wiki_*.bz2 files
        limit: Maximum number of articles to process (0 = no limit)

    Returns:
        List of Document objects ready for Chroma indexing
    """
    all_docs = []
    article_count = 0

    # Find all bz2 files
    bz2_files = sorted(wiki_dir.glob("**/wiki_*.bz2"))
    print(f"[INFO] Found {len(bz2_files)} bz2 files in {wiki_dir}")

    for bz2_path in bz2_files:
        try:
            with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        article = json.loads(line)
                        docs = process_wiki_article(article)
                        all_docs.extend(docs)

                        if docs:
                            article_count += 1
                            if article_count % 1000 == 0:
                                print(f"[PROGRESS] Processed {article_count} articles -> {len(all_docs)} documents")

                        # Check limit
                        if limit > 0 and article_count >= limit:
                            print(f"[INFO] Reached limit of {limit} articles")
                            return all_docs

                    except json.JSONDecodeError as e:
                        print(f"[WARN] JSON decode error in {bz2_path}:{line_num}: {e}")
                        continue

        except Exception as e:
            print(f"[ERROR] Failed to process {bz2_path}: {e}")
            continue

    print(f"[INFO] Total: {article_count} articles -> {len(all_docs)} documents")
    return all_docs


def index_documents(
    docs: List[Document],
    persist_dir: str,
    collection_name: str,
    batch_size: int = 500
):
    """
    Index documents into Chroma with batching for memory efficiency.
    """
    print(f"[INFO] Indexing {len(docs)} documents into collection '{collection_name}'")
    print(f"[INFO] Persist directory: {persist_dir}")

    # Initialize embedding model using centralized configuration
    # To use a different model, set EMBEDDING_MODEL_PATH environment variable:
    # export EMBEDDING_MODEL_PATH="/path/to/your/model"
    from core.config import EmbeddingConfig
    embedding = EmbeddingConfig.get_embedding(device="cuda")

    # Create or load Chroma collection
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding,
    )

    # Batch indexing
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"[PROGRESS] Indexed batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} ({i+len(batch)}/{len(docs)} docs)")

    # Verify count
    collection = vectorstore._collection
    final_count = collection.count()
    print(f"[SUCCESS] Total documents in collection: {final_count}")

    # Sample random document
    print("\n[SAMPLE] Random document from collection:")
    sample = collection.peek(limit=1)
    if sample and sample.get("documents"):
        print(f"  Content preview: {sample['documents'][0][:300]}...")
        if sample.get("metadatas"):
            print(f"  Metadata: {sample['metadatas'][0]}")

    return vectorstore


def test_retrieval(vectorstore: Chroma, test_query: str = "Who directed Doctor Strange?"):
    """
    Test retrieval with a sample query.
    """
    print(f"\n[TEST] Testing retrieval with query: '{test_query}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(test_query)

    print(f"[TEST] Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("title", "N/A")
        num_sents = doc.metadata.get("num_sentences", "N/A")
        content_preview = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {title} ({num_sents} sentences) - {content_preview}...")


def main():
    ap = argparse.ArgumentParser(
        description="Index HotpotQA Full Wiki into Chroma for retrieval"
    )
    ap.add_argument(
        "--wiki-dir",
        type=str,
        required=True,
        help="Path to enwiki-20171001-pages-meta-current-withlinks-abstracts directory"
    )
    ap.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db_hotpotqa_fullwiki",
        help="Chroma persist directory (default: ./chroma_db_hotpotqa_fullwiki)"
    )
    ap.add_argument(
        "--collection",
        type=str,
        default="hotpotqa_fullwiki",
        help="Chroma collection name (default: hotpotqa_fullwiki)"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of articles to process (0 = no limit, default: 0)"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for indexing (default: 500)"
    )
    ap.add_argument(
        "--test-query",
        type=str,
        default="Who directed Doctor Strange?",
        help="Test query for retrieval verification"
    )

    args = ap.parse_args()

    wiki_dir = Path(args.wiki_dir)
    if not wiki_dir.exists():
        print(f"[ERROR] Wiki directory not found: {wiki_dir}")
        sys.exit(1)

    print("="*60)
    print("HotpotQA Full Wiki Indexing")
    print("="*60)
    print(f"Wiki directory: {wiki_dir}")
    print(f"Persist directory: {args.persist_dir}")
    print(f"Collection name: {args.collection}")
    print(f"Article limit: {args.limit if args.limit > 0 else 'unlimited'}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)

    # Load documents
    print("\n[STEP 1/3] Loading Wikipedia articles...")
    docs = load_wiki_articles(wiki_dir, limit=args.limit)

    if not docs:
        print("[ERROR] No documents loaded! Check your wiki directory.")
        sys.exit(1)

    # Index documents
    print(f"\n[STEP 2/3] Indexing {len(docs)} documents...")
    vectorstore = index_documents(
        docs=docs,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size
    )

    # Test retrieval
    print("\n[STEP 3/3] Testing retrieval...")
    test_retrieval(vectorstore, test_query=args.test_query)

    print("\n" + "="*60)
    print("[DONE] Indexing complete!")
    print(f"  Total documents: {len(docs)}")
    print(f"  Collection: {args.collection}")
    print(f"  Persist dir: {args.persist_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
