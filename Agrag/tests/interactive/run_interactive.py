import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 导入 core.config 会自动加载 .env 文件
from core.config import PERSIST_DIR, COLLECTION_NAME
from runner.logging import C
from runner.engine import (
    build_app,
    run_one,
    save_workflow_png,
)

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Interactive Runner")
    parser.add_argument(
        "--rag",
        type=str,
        default="crag",
        choices=["agrag", "crag", "hop2rag", "selfrag"],
        help="选择运行的 RAG 工作流（默认 crag）",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save workflow graph as PNG (or .mmd fallback)",
    )
    args = parser.parse_args()

    # 使用 .env 中配置的路径
    persist_dir = PERSIST_DIR
    collection_name = COLLECTION_NAME

    print(f"\n{C.BOLD}====== Agentic RAG | vLLM Backend ======{C.END}")
    print(f"{C.GRAY}KB: {persist_dir}{C.END}")
    print(f"{C.GRAY}Collection: {collection_name}{C.END}")
    print(f"{C.GRAY}RAG: {args.rag}{C.END}")

    print(f"{C.GRAY}Type 'exit' or 'quit' to leave.{C.END}\n")

    thread_id = input("Enter user ID (press Enter for default): ").strip()
    if not thread_id:
        thread_id = f"usr_{int(time.time())}"

    handle = build_app(rag_name=args.rag)

    if args.save_graph:
        path = save_workflow_png(handle, output_file="agent_workflow.png")
        print(f"{C.GREEN}[VISUALIZE] Workflow graph saved to: {path}{C.END}")

    custom_retriever_config = {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "k": "10",
    }

    while True:
        try:
            question = input(f"\n{C.BOLD}User Input: {C.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            out = run_one(
                handle,
                question=question,
                thread_id=thread_id,
                custom_retriever_config=custom_retriever_config,
            )

            print(f"\n{C.BOLD}Assistant: {C.END} {out['prediction']}")
            print(
                f"{C.GRAY}"
                f"[Source={out['data_source']}, "
                f"Router={out['router_decision']}, "
                f"Useful={out['useful']}]"
                f"{C.END}"
            )

        except Exception as e:
            print(f"{C.RED}Error: {e}{C.END}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
