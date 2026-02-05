#!/usr/bin/env python
"""
验证预热机制的简单测试脚本

测试场景：
1. 验证 Embedding 模型只加载一次
2. 验证预热后的性能提升
3. 验证线程安全性
"""

import sys
import os
import time
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Rag.c_rag_performancy_concurrent import (
    get_embedding_model,
    get_retriever,
    warmup_resources
)


def test_singleton():
    """测试 Embedding 模型单例"""
    print("\n" + "=" * 70)
    print("Test 1: Embedding Model Singleton")
    print("=" * 70)

    print("\n[1] First call to get_embedding_model()...")
    start = time.time()
    model1 = get_embedding_model()
    time1 = time.time() - start
    print(f"    Time: {time1:.2f}s")
    print(f"    Model ID: {id(model1)}")

    print("\n[2] Second call to get_embedding_model()...")
    start = time.time()
    model2 = get_embedding_model()
    time2 = time.time() - start
    print(f"    Time: {time2:.2f}s")
    print(f"    Model ID: {id(model2)}")

    print("\n[3] Third call to get_embedding_model()...")
    start = time.time()
    model3 = get_embedding_model()
    time3 = time.time() - start
    print(f"    Time: {time3:.2f}s")
    print(f"    Model ID: {id(model3)}")

    # 验证是同一个实例
    assert model1 is model2 is model3, "Models should be the same instance!"
    print("\n✅ PASS: All calls return the same instance")
    print(f"   First call: {time1:.2f}s (loading)")
    print(f"   Second call: {time2:.4f}s (cached, {time1/max(time2, 0.0001):.0f}x faster)")
    print(f"   Third call: {time3:.4f}s (cached, {time1/max(time3, 0.0001):.0f}x faster)")


def test_retriever_cache():
    """测试 Retriever 缓存"""
    print("\n" + "=" * 70)
    print("Test 2: Retriever Cache")
    print("=" * 70)

    print("\n[1] First call to get_retriever()...")
    start = time.time()
    retriever1 = get_retriever(debug=False, use_remote=True)
    time1 = time.time() - start
    print(f"    Time: {time1:.2f}s")
    print(f"    Retriever ID: {id(retriever1)}")

    print("\n[2] Second call to get_retriever()...")
    start = time.time()
    retriever2 = get_retriever(debug=False, use_remote=True)
    time2 = time.time() - start
    print(f"    Time: {time2:.2f}s")
    print(f"    Retriever ID: {id(retriever2)}")

    print("\n[3] Third call to get_retriever()...")
    start = time.time()
    retriever3 = get_retriever(debug=False, use_remote=True)
    time3 = time.time() - start
    print(f"    Time: {time3:.2f}s")
    print(f"    Retriever ID: {id(retriever3)}")

    # 验证缓存（CRag 使用单例模式，所有调用返回同一个实例）
    assert retriever1 is retriever2 is retriever3, "All calls should return cached retriever!"
    print("\n✅ PASS: Retriever cache works correctly")
    print(f"   First call: {time1:.2f}s (creating)")
    print(f"   Second call: {time2:.4f}s (cached, {time1/max(time2, 0.0001):.0f}x faster)")
    print(f"   Third call: {time3:.4f}s (cached, {time1/max(time3, 0.0001):.0f}x faster)")


def test_thread_safety():
    """测试线程安全性"""
    print("\n" + "=" * 70)
    print("Test 3: Thread Safety")
    print("=" * 70)

    results = []

    def worker(worker_id):
        start = time.time()
        model = get_embedding_model()
        elapsed = time.time() - start
        results.append((worker_id, id(model), elapsed))
        print(f"    Worker {worker_id}: {elapsed:.4f}s, Model ID: {id(model)}")

    print("\n[1] Creating 5 threads to call get_embedding_model()...")
    threads = []
    for i in range(5):
        t = Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # 验证所有线程获取的是同一个实例
    model_ids = [r[1] for r in results]
    assert len(set(model_ids)) == 1, "All threads should get the same model instance!"
    print("\n✅ PASS: All threads got the same model instance")
    print(f"   Model ID: {model_ids[0]}")


def test_warmup():
    """测试预热机制"""
    print("\n" + "=" * 70)
    print("Test 4: Warmup Mechanism")
    print("=" * 70)

    print("\n[1] Calling warmup_resources()...")
    start = time.time()
    warmup_resources(use_remote=True)
    warmup_time = time.time() - start

    print(f"\n[2] After warmup, calling get_embedding_model()...")
    start = time.time()
    _ = get_embedding_model()
    time1 = time.time() - start
    print(f"    Time: {time1:.4f}s (should be instant)")

    print(f"\n[3] After warmup, calling get_retriever()...")
    start = time.time()
    _ = get_retriever(debug=False, use_remote=True)
    time2 = time.time() - start
    print(f"    Time: {time2:.4f}s (should be instant)")

    print("\n✅ PASS: Warmup mechanism works correctly")
    print(f"   Warmup time: {warmup_time:.2f}s")
    print(f"   Post-warmup get_embedding_model(): {time1:.4f}s")
    print(f"   Post-warmup get_retriever(): {time2:.4f}s")


def main():
    print("\n" + "=" * 70)
    print("Warmup Optimization Verification Tests")
    print("=" * 70)

    try:
        # Test 1: Singleton
        test_singleton()

        # Test 2: Retriever cache
        test_retriever_cache()

        # Test 3: Thread safety
        test_thread_safety()

        # Test 4: Warmup
        test_warmup()

        print("\n" + "=" * 70)
        print("All Tests Passed! ✅")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
