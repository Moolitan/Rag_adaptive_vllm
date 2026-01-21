#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Iterator
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm


def load_wiki_passages_stream(
    tsv_file: Path,
    limit: int = 0,
    skip_lines: int = 0
) -> Iterator[Document]:
    """
    流式读取 DPR Wikipedia passages TSV 文件，逐行生成 Document 对象。

    Args:
        tsv_file: TSV 文件路径
        limit: 限制读取行数（0=全部）
        skip_lines: 跳过前 N 行（用于断点续传）

    Yields:
        Document 对象
    """
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        # 跳过指定行数
        for _ in range(skip_lines):
            try:
                next(reader)
            except StopIteration:
                return

        count = 0
        for row in reader:
            try:
                doc_id = row.get('id', '').strip()
                text = row.get('text', '').strip()
                title = row.get('title', '').strip()

                # 过滤空文档
                if not text or len(text) < 10:
                    continue

                doc = Document(
                    page_content="passage: " + text,
                    metadata={
                        'id': doc_id,
                        'title': title,
                        'source': 'wikipedia_dpr'
                    }
                )

                yield doc
                count += 1

                if limit > 0 and count >= limit:
                    break

            except Exception as e:
                # 忽略格式错误的行
                continue


def estimate_file_lines(tsv_file: Path) -> int:
    """
    估算 TSV 文件行数（快速采样估算）
    """
    import subprocess
    try:
        result = subprocess.run(
            ['wc', '-l', str(tsv_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0]) - 1  # 减去表头
    except Exception:
        pass

    # 回退：采样估算
    sample_size = 10000
    with open(tsv_file, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        # 读取样本计算平均行大小
        sample_bytes = sum(len(line.encode('utf-8')) for _, line in zip(range(sample_size), f))
        avg_line_size = sample_bytes / sample_size

    file_size = tsv_file.stat().st_size
    return int(file_size / avg_line_size) - 1


def create_faiss_index_incremental(
    tsv_file: Path,
    embedding: HuggingFaceEmbeddings,
    batch_size: int,
    limit: int = 0,
    use_gpu: bool = False,
    checkpoint_dir: Path = None
) -> FAISS:
    """
    增量创建 FAISS 索引，支持大规模数据和断点续传

    Args:
        tsv_file: TSV 文件路径
        embedding: 嵌入模型
        batch_size: 批量大小
        limit: 限制行数
        use_gpu: 是否使用 GPU
        checkpoint_dir: 检查点目录（用于断点续传）

    Returns:
        FAISS vectorstore
    """
    # 估算总行数
    if limit > 0:
        total_docs = limit
    else:
        print("Estimating total documents...")
        total_docs = estimate_file_lines(tsv_file)
        print(f"Estimated total: ~{total_docs:,} documents")

    # 检查是否有检查点
    start_batch = 0
    vectorstore = None

    if checkpoint_dir and checkpoint_dir.exists():
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_batch = checkpoint.get('batch_num', 0)
                print(f"Resuming from checkpoint: batch {start_batch}")

            # 加载已有索引
            try:
                vectorstore = FAISS.load_local(
                    str(checkpoint_dir),
                    embedding,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing index with {vectorstore.index.ntotal} documents")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                start_batch = 0
                vectorstore = None

    # 流式处理
    doc_stream = load_wiki_passages_stream(
        tsv_file,
        limit=limit,
        skip_lines=start_batch * batch_size
    )

    batch = []
    batch_num = start_batch
    indexed_count = vectorstore.index.ntotal if vectorstore else 0

    pbar = tqdm(
        total=total_docs,
        initial=indexed_count,
        desc="Indexing",
        unit="docs"
    )

    start_time = time.time()

    for doc in doc_stream:
        batch.append(doc)

        if len(batch) >= batch_size:
            # 处理当前批次
            try:
                if vectorstore is None:
                    # 初始化索引
                    vectorstore = FAISS.from_documents(batch, embedding)
                    print(f"\n✓ Initialized index with {len(batch)} documents")
                else:
                    # 添加到已有索引
                    vectorstore.add_documents(batch)

                indexed_count += len(batch)
                batch_num += 1
                pbar.update(len(batch))

                # 保存检查点
                if checkpoint_dir and batch_num % 10 == 0:  # 每10个batch保存一次
                    vectorstore.save_local(str(checkpoint_dir))
                    with open(checkpoint_dir / "checkpoint.json", 'w') as f:
                        json.dump({
                            'batch_num': batch_num,
                            'indexed_count': indexed_count,
                            'timestamp': time.time()
                        }, f)

                # 速度统计
                elapsed = time.time() - start_time
                docs_per_sec = indexed_count / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'batch': batch_num,
                    'speed': f'{docs_per_sec:.0f} docs/s'
                })

            except Exception as e:
                print(f"\nError processing batch {batch_num}: {e}")
                # 保存当前进度
                if checkpoint_dir and vectorstore:
                    vectorstore.save_local(str(checkpoint_dir))
                raise

            batch = []

    # 处理最后一个不满的batch
    if batch and vectorstore:
        vectorstore.add_documents(batch)
        indexed_count += len(batch)
        pbar.update(len(batch))

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\n✓ Indexed {indexed_count:,} documents in {elapsed:.1f}s")
    print(f"  Average speed: {indexed_count / elapsed:.0f} docs/s")

    return vectorstore


def main():
    ap = argparse.ArgumentParser(
        description="Index DPR Wikipedia passages into FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 100k documents (CPU)
  python data/index_wiki_faiss.py --limit 100000 --batch-size 5000

  # Full index with GPU acceleration
  python data/index_wiki_faiss.py --use-gpu --batch-size 50000

  # Resume interrupted indexing
  python data/index_wiki_faiss.py --use-gpu --resume
        """
    )

    ap.add_argument(
        "--tsv-file",
        type=str,
        default="data/wiki/psgs_w100.tsv",
        help="Path to DPR Wikipedia passages TSV file"
    )
    ap.add_argument(
        "--index-dir",
        type=str,
        default="/mnt/Large_Language_Model_Lab_1/faiss_index_wiki_dpr",
        help="Output directory for FAISS index"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents (0=all, ~21M)"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for indexing (larger=faster but more memory)"
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for embedding model (requires CUDA)"
    )
    ap.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model name or local path (e.g., BAAI/bge-base-en-v1.5)"
    )
    ap.add_argument(
        "--encode-batch-size",
        type=int,
        default=0,
        help="Embedding encode batch size (0=auto: 512 for GPU, 128 for CPU)"
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    args = ap.parse_args()

    tsv_file = Path(args.tsv_file)
    index_dir = Path(args.index_dir)

    # 检查输入文件
    if not tsv_file.exists():
        print(f"Error: TSV file not found: {tsv_file}")
        print(f"\nDownload it with:")
        print(f"  wget -c https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz")
        print(f"  gunzip psgs_w100.tsv.gz")
        sys.exit(1)

    # 创建输出目录
    index_dir.mkdir(parents=True, exist_ok=True)

    # 检查点目录
    checkpoint_dir = index_dir if args.resume else None

    print("=" * 80)
    print("FAISS Indexing - Wikipedia DPR Passages")
    print("=" * 80)
    print(f"TSV file:        {tsv_file} ({tsv_file.stat().st_size / 1e9:.1f} GB)")
    print(f"Index directory: {index_dir}")
    print(f"Batch size:      {args.batch_size:,} (documents per FAISS batch)")
    encode_bs_display = args.encode_batch_size if args.encode_batch_size > 0 else "auto"
    print(f"Encode batch:    {encode_bs_display} (embedding model internal batch)")
    print(f"Limit:           {args.limit:,} documents" if args.limit > 0 else "Limit:           All documents (~21M)")
    print(f"GPU mode:        {args.use_gpu}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Resume mode:     {args.resume}")
    print("=" * 80)
    print()

    # 初始化嵌入模型
    print("Loading embedding model...")
    device = 'cuda' if args.use_gpu else 'cpu'

    # 统一的模型存储目录
    model_base_dir = Path("/mnt/Large_Language_Model_Lab_1/模型/rag_models")
    model_base_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量，让 sentence-transformers 使用指定的缓存目录
    import os
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(model_base_dir)
    os.environ['HF_HOME'] = str(model_base_dir)

    # 构造本地模型路径（将 / 替换为 -）
    model_local_name = args.embedding_model.replace('/', '-')
    local_model_path = model_base_dir / model_local_name

    # 检查本地是否已有模型
    if local_model_path.exists() and local_model_path.is_dir():
        # 本地模型存在，直接使用
        print(f"Found local model: {local_model_path}")
        model_name_or_path = str(local_model_path)
    else:
        # 本地不存在，从 HuggingFace Hub 下载
        print(f"Model not found locally: {local_model_path}")
        print(f"Downloading from HuggingFace Hub: {args.embedding_model}")
        print(f"This may take a few minutes...")
        print()

        # 使用 HuggingFace 模型名，它会自动下载到缓存目录
        model_name_or_path = args.embedding_model

    try:
        # 设置embedding批量大小（关键性能参数）
        if args.encode_batch_size > 0:
            encode_batch_size = args.encode_batch_size
        else:
            # 自动设置：GPU用更大的batch，CPU用较小的
            encode_batch_size = 512 if args.use_gpu else 128

        embedding = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': encode_batch_size,  # 内部批量处理大小
            },
        )
        print(f"✓ Embedding model loaded on {device}")
        print(f"✓ Encode batch size: {encode_batch_size}")

        # 如果是从 HuggingFace 下载的，提示下次会从本地加载
        if not local_model_path.exists():
            print(f"Note: Model downloaded to cache. To use as local model,")
            print(f"      copy/symlink it to: {local_model_path}")

        # 测试嵌入
        test_emb = embedding.embed_query("test")
        print(f"✓ Embedding dimension: {len(test_emb)}")
        print()

    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Model name: {args.embedding_model}")
        print(f"  2. Local path checked: {local_model_path}")
        print(f"  3. Cache directory: {model_base_dir}")
        print(f"  4. Check internet connection for HuggingFace download")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 创建 FAISS 索引
    print("Creating FAISS index...")
    print(f"Processing in batches of {args.batch_size:,}...")
    print()

    try:
        vectorstore = create_faiss_index_incremental(
            tsv_file=tsv_file,
            embedding=embedding,
            batch_size=args.batch_size,
            limit=args.limit,
            use_gpu=args.use_gpu,
            checkpoint_dir=checkpoint_dir
        )

    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user")
        print(f"Progress saved to: {index_dir}")
        print(f"Resume with: python {sys.argv[0]} --resume")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during indexing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 保存最终索引
    print(f"\nSaving FAISS index to {index_dir}...")
    vectorstore.save_local(str(index_dir))
    print(f"✓ Index saved")
    print()

    # 保存元数据
    metadata = {
        "num_documents": vectorstore.index.ntotal,
        "embedding_model": args.embedding_model,
        "embedding_dim": len(test_emb),
        "gpu_mode": args.use_gpu,
        "batch_size": args.batch_size,
        "source_file": str(tsv_file),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(index_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved")
    print()

    # 测试检索
    print("Testing retrieval...")
    results = vectorstore.similarity_search("query: What is artificial intelligence?", k=3)
    print(f"✓ Retrieved {len(results)} documents")
    print(f"\nSample result:")
    print(f"  Title: {results[0].metadata.get('title', 'N/A')}")
    print(f"  Text:  {results[0].page_content[:200]}...")
    print()

    print("=" * 80)
    print("Indexing Complete!")
    print("=" * 80)
    print(f"Index location:  {index_dir}")
    print(f"Total documents: {vectorstore.index.ntotal:,}")
    print(f"Index size:      {sum(f.stat().st_size for f in index_dir.glob('*')) / 1e9:.2f} GB")
    print()
    print("To use this index in your RAG system:")
    print(f'  export AGRAG_FAISS_INDEX_DIR="{index_dir}"')
    print()
    print("Test retrieval:")
    print(f'  python -c "')
    print(f'from langchain_community.vectorstores import FAISS')
    print(f'from langchain_huggingface import HuggingFaceEmbeddings')
    print(f'embedding = HuggingFaceEmbeddings(model_name=\\"{args.embedding_model}\\")')
    print(f'vs = FAISS.load_local(\\"{index_dir}\\", embedding, allow_dangerous_deserialization=True)')
    print(f'results = vs.similarity_search(\\"artificial intelligence\\", k=5)')
    print(f'print(results[0])')
    print(f'  "')
    print()


if __name__ == "__main__":
    main()
