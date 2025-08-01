#!/usr/bin/env python3
"""
DynaGraph Integration Demo Script

这个脚本演示了DynaGraph已经成功集成到SAGE-DB-Bench中，
可以与其他算法一起进行benchmark测试。
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append('DynaGraph/bindings/build')
sys.path.append('big-ann-benchmarks')

print("=" * 60)
print("DynaGraph Integration Demo")
print("=" * 60)

try:
    from neurips23.streaming.dynagraph.dynagraph import DynaGraph
    print("✓ DynaGraph streaming adapter imported successfully")
    
    # 测试不同的 batch_size 设置
    batch_sizes_to_test = [500, 1000, 2000, 5000]
    n_points = 3000  # 固定测试点数
    
    for batch_size in batch_sizes_to_test:
        print(f"\n--- Testing with batch_size = {batch_size} ---")
        
        # 创建算法实例
        dg = DynaGraph('euclidean', {
            'alpha': 1.2, 
            'coef_L': 30, 
            'coef_R': 12,
            'batch_size': batch_size
        })
        print(f"✓ Algorithm name: {dg.name}")
        
        # 设置索引
        n_dim = 128
        max_pts = 10000
        dg.setup(np.float32, max_pts, n_dim)
        print(f"✓ Index setup: {n_dim}D, max {max_pts} points")
        
        # 插入测试数据
        X = np.random.random((n_points, n_dim)).astype(np.float32)
        ids = np.arange(n_points, dtype=np.uint64)
        
        print(f"✓ Inserting {n_points} points with batch_size={batch_size}...")
        dg.insert(X, ids)
        print("✓ Insert completed")
        
        # 测试查询
        print(f"--- Testing Query Performance (batch_size={batch_size}) ---")
        
        # 选择一些插入的点作为查询
        test_indices = np.random.choice(n_points, 3, replace=False)
        test_queries = X[test_indices]
        k = 10
        
        print(f"✓ Running test with {len(test_indices)} queries, k={k}")
        
        # DynaGraph查询结果
        dg_results = dg.query(test_queries, k)
        print(f"✓ DynaGraph query completed, results shape: {dg_results.shape}")
        
        # 检查查询结果
        valid_queries = 0
        for i, result in enumerate(dg_results):
            non_zero_count = np.count_nonzero(result)
            print(f"  Query {i}: {non_zero_count} non-zero results")
            if non_zero_count > 0:
                valid_queries += 1
        
        print(f"✓ Valid queries: {valid_queries}/{len(test_indices)}")
        
        if valid_queries > 0:
            print(f"✓ SUCCESS: batch_size={batch_size} returns valid results!")
        else:
            print(f"✗ FAILED: batch_size={batch_size} returns no valid results")
    
    print("\n" + "=" * 60)
    print("Batch size analysis completed!")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
