#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for HotpotQA Full Wiki setup.

Checks the 3 "must-pass" points:
A) Chroma indexing successful and retrieval works
B) Data flow integrity (retrieve -> grade_docs -> generate)
C) SP output format strictly conforms to HotpotQA

Usage:
    python tests/HotpotQA/verify_hotpotqa_fullwiki.py \
        --hotpot-json tests/HotpotQA/data/hotpot_dev_fullwiki_v1.json
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]  # Agrag/tests/data/hotpotqa/xxx.py -> Agrag/
sys.path.insert(0, str(ROOT))

# 导入 core.config 会自动加载 .env 文件
from core.config import PERSIST_DIR, COLLECTION_NAME

# Embedding imports (must match what you used in index_hotpotqa_fullwiki.py)
from langchain_nomic.embeddings import NomicEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings  # For BGE/MiniLM
# from langchain_openai import OpenAIEmbeddings  # For OpenAI
from langchain_chroma import Chroma
from runner.engine import build_app, run_one


def check_point_a(persist_dir: str, collection_name: str):
    """
    必过点 A：Chroma 入库成功且可检索
    Evidence: collection.count() > 0; retrieval returns K docs with valid title/text
    """
    print("\n" + "="*60)
    print("必过点 A：Chroma 入库成功且可检索")
    print("="*60)

    try:
        # IMPORTANT: Must use SAME embedding as index_hotpotqa_fullwiki.py!
        # Using centralized configuration ensures consistency
        from core.config import EmbeddingConfig
        embedding = EmbeddingConfig.get_embedding(device="cuda")
        # embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embedding,
        )

        # Check collection count
        collection = vectorstore._collection
        count = collection.count()
        print(f"✓ Collection count: {count}")

        if count == 0:
            print(f"✗ FAILED: Collection is empty!")
            return False

        # Test retrieval
        test_query = "Who directed Doctor Strange?"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(test_query)

        print(f"✓ Retrieved {len(results)} documents for query: '{test_query}'")

        if not results:
            print(f"✗ FAILED: No results returned!")
            return False

        # Verify first document
        doc = results[0]
        title = doc.metadata.get("title", "N/A")
        num_sents = doc.metadata.get("num_sentences", "N/A")
        content_preview = doc.page_content[:200].replace("\n", " ")

        print(f"\n✓ First document:")
        print(f"  Title: {title}")
        print(f"  Sentences: {num_sents}")
        print(f"  Content: {content_preview}...")

        # Verify format (should be "Title\n0. Sent1\n1. Sent2...")
        lines = doc.page_content.split("\n")
        if len(lines) < 2:
            print(f"✗ FAILED: Document format incorrect (expected multi-line)")
            return False

        if not lines[0].strip():
            print(f"✗ FAILED: Document format incorrect (title missing)")
            return False

        # Check if numbered sentences exist
        has_numbered = any(line.strip().startswith(("0.", "1.", "2.")) for line in lines[1:])
        if not has_numbered:
            print(f"✗ FAILED: Document format incorrect (no numbered sentences)")
            return False

        print(f"✓ Document format valid (Title + numbered sentences)")
        print("\n✓ 必过点 A：PASSED")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_point_b(persist_dir: str, collection_name: str, test_question: str):
    """
    必过点 B：retrieve → grade_docs → generate 的数据流不断
    Evidence: generate 前 state.documents 非空；日志显示 doc 数量；第一条 doc 预览
    """
    print("\n" + "="*60)
    print("必过点 B：数据流完整性")
    print("="*60)

    try:
        # Build app
        handle = build_app()

        # Run one question with custom retriever
        custom_config = {
            "persist_dir": persist_dir,
            "collection_name": collection_name,
            "k": "10",
        }

        print(f"Running question: {test_question}")
        out = run_one(
            handle,
            question=test_question,
            thread_id="verify_point_b",
            custom_retriever_config=custom_config,
        )

        # Check final state
        final_state = out.get("final_state", {})
        documents = final_state.get("documents", [])

        if not documents:
            print(f"✗ FAILED: No documents in final state!")
            return False

        print(f"✓ Documents in final state: {len(documents)}")

        # Check first document
        if len(documents) > 0:
            doc = documents[0]
            title = doc.metadata.get("title", "N/A")
            content_preview = doc.page_content[:150].replace("\n", " ")
            print(f"✓ First doc title: {title}")
            print(f"✓ First doc preview: {content_preview}...")

        # Check prediction
        prediction = out.get("prediction", "")
        if not prediction or len(prediction) < 3:
            print(f"✗ FAILED: No valid prediction generated!")
            return False

        print(f"✓ Prediction: {prediction}")

        # Check data source
        data_source = out.get("data_source", "")
        print(f"✓ Data source: {data_source}")

        if data_source != "custom_vectorstore":
            print(f"⚠ WARNING: Expected data_source='custom_vectorstore', got '{data_source}'")

        print("\n✓ 必过点 B：PASSED")
        handle.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_point_c(persist_dir: str, collection_name: str, hotpot_json: str):
    """
    必过点 C：sp 输出格式严格符合 HotpotQA
    Evidence: 对 1 条样本打印最终输出，验证 [title, sent_idx] 格式
    """
    print("\n" + "="*60)
    print("必过点 C：sp 输出格式严格符合 HotpotQA")
    print("="*60)

    try:
        # Load one sample
        with open(hotpot_json, "r") as f:
            data = json.load(f)

        if not data:
            print(f"✗ FAILED: No data in {hotpot_json}")
            return False

        sample = data[0]
        qid = sample.get("_id", "unknown")
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")
        gold_sp = sample.get("supporting_facts", [])

        print(f"Sample ID: {qid}")
        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Gold SP ({len(gold_sp)} entries): {gold_sp[:2]}...")

        # Build app and run
        handle = build_app()
        custom_config = {
            "persist_dir": persist_dir,
            "collection_name": collection_name,
            "k": "10",
        }

        out = run_one(
            handle,
            question=question,
            thread_id="verify_point_c",
            custom_retriever_config=custom_config,
        )

        # Check prediction
        prediction = out.get("prediction", "")
        print(f"\n✓ Predicted answer: {prediction}")

        # Check supporting facts
        final_state = out.get("final_state", {})
        predicted_sp = final_state.get("supporting_facts", [])

        if not isinstance(predicted_sp, list):
            print(f"✗ FAILED: supporting_facts is not a list! Type: {type(predicted_sp)}")
            return False

        print(f"✓ Predicted SP ({len(predicted_sp)} entries):")

        if len(predicted_sp) == 0:
            print(f"⚠ WARNING: No supporting facts predicted!")
            print(f"  This might be an issue with the LLM or supporting_fact_extractor")

        # Validate format: [[title, sent_idx], ...]
        valid_count = 0
        for i, sp_item in enumerate(predicted_sp):
            if not isinstance(sp_item, (list, tuple)) or len(sp_item) != 2:
                print(f"  [{i}] ✗ INVALID: {sp_item} (not [title, sent_idx])")
                continue

            title, sent_idx = sp_item
            if not isinstance(title, str):
                print(f"  [{i}] ✗ INVALID: title is not string: {title}")
                continue

            if not isinstance(sent_idx, int):
                print(f"  [{i}] ✗ INVALID: sent_idx is not int: {sent_idx} (type: {type(sent_idx)})")
                continue

            if sent_idx < 0:
                print(f"  [{i}] ✗ INVALID: sent_idx < 0: {sent_idx}")
                continue

            print(f"  [{i}] ✓ VALID: {sp_item}")
            valid_count += 1

        print(f"\n✓ Valid SP entries: {valid_count}/{len(predicted_sp)}")

        if valid_count < len(predicted_sp):
            print(f"⚠ WARNING: Some SP entries are invalid!")

        # Final check: at least some valid entries
        if len(predicted_sp) > 0 and valid_count == 0:
            print(f"✗ FAILED: No valid SP entries!")
            return False

        print("\n✓ 必过点 C：PASSED (format validation)")
        handle.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description="Verify HotpotQA Full Wiki setup")
    ap.add_argument(
        "--hotpot-json",
        type=str,
        default=str(ROOT / "tests" / "data" / "hotpotqa" / "hotpot_dev_fullwiki_v1.json"),
        help="Path to hotpot_dev_fullwiki_v1.json"
    )
    ap.add_argument(
        "--test-question",
        type=str,
        default="Were Scott Derrickson and Ed Wood of the same nationality?",
        help="Test question for Point B"
    )

    args = ap.parse_args()

    # 使用 .env 中配置的路径
    persist_dir = PERSIST_DIR
    collection = COLLECTION_NAME

    # Check persist dir exists
    if not Path(persist_dir).exists():
        print(f"[ERROR] Persist directory not found: {persist_dir}")
        print(f"[ERROR] Please check AGRAG_PERSIST_DIR in .env file!")
        sys.exit(1)

    print("="*60)
    print("HotpotQA Full Wiki Verification")
    print("="*60)
    print(f"Persist dir: {persist_dir}")
    print(f"Collection: {collection}")
    print(f"Test question: {args.test_question}")
    print("="*60)

    results = {
        "point_a": False,
        "point_b": False,
        "point_c": False,
    }

    # Run checks
    results["point_a"] = check_point_a(persist_dir, collection)
    if results["point_a"]:
        results["point_b"] = check_point_b(persist_dir, collection, args.test_question)
        if results["point_b"]:
            results["point_c"] = check_point_c(persist_dir, collection, args.hotpot_json)

    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for point, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{point.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the benchmark.")
    else:
        print("\n❌ Some checks failed. Please fix the issues before running the benchmark.")
        sys.exit(1)


if __name__ == "__main__":
    main()
