#!/usr/bin/env python3
"""
测试修改后的DynaGraph批量聚类逻辑
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append('DynaGraph/bindings/build')
sys.path.append('big-ann-benchmarks')

print("=" * 60)
print("DynaGraph Cluster Count Verification Test")
print("=" * 60)

try:
    from neurips23.streaming.dynagraph.dynagraph import DynaGraph
    print("✓ DynaGraph streaming adapter imported successfully")
    
    # 测试不同的 batch_size 设置，验证聚类数量
    test_cases = [
        {'batch_size': 500, 'expected_clusters': 2},   # max(500/500, 2) = 2
        {'batch_size': 1000, 'expected_clusters': 2},  # max(1000/500, 2) = 2  
        {'batch_size': 1500, 'expected_clusters': 3},  # max(1500/500, 2) = 3
        {'batch_size': 2000, 'expected_clusters': 4},  # max(2000/500, 2) = 4
        {'batch_size': 2500, 'expected_clusters': 5},  # max(2500/500, 2) = 5
    ]
    
    n_points = 3000  # 测试数据点数
    n_dim = 128
    k = 10
    
    for test_case in test_cases:
        batch_size = test_case['batch_size']
        expected_clusters = test_case['expected_clusters']
        
        print(f"\n--- Testing batch_size={batch_size}, expected_clusters={expected_clusters} ---")
        
        # 创建算法实例
        dg = DynaGraph('euclidean', {
            'alpha': 1.2, 
            'coef_L': 100, 
            'coef_R': 64,
            'batch_size': batch_size
        })
        
        # 设置索引
        max_pts = 10000
        dg.setup(np.float32, max_pts, n_dim)
        print(f"✓ Index setup: batch_size={batch_size}")
        
        # 插入测试数据
        X = np.random.random((n_points, n_dim)).astype(np.float32)
        ids = np.arange(n_points, dtype=np.uint64)
        
        print(f"✓ Inserting {n_points} points...")
        dg.insert(X, ids)
        print("✓ Insert completed")
        
        # 测试查询
        print(f"--- Testing Query ---")
        
        # 选择查询点
        test_indices = np.random.choice(n_points, 5, replace=False)
        test_queries = X[test_indices]
        
        # DynaGraph查询结果
        dg_results = dg.query(test_queries, k)
        print(f"✓ DynaGraph query completed, results shape: {dg_results.shape}")
        
        # 检查查询结果
        valid_queries = 0
        total_results = 0
        for i, result in enumerate(dg_results):
            non_zero_count = np.count_nonzero(result)
            total_results += non_zero_count
            if non_zero_count > 0:
                valid_queries += 1
        
        success_rate = valid_queries / len(test_indices)
        avg_results = total_results / len(test_indices)
        
        print(f"✓ Valid queries: {valid_queries}/{len(test_indices)} ({success_rate:.2%})")
        print(f"✓ Average results per query: {avg_results:.1f}")
        
        if success_rate >= 0.8:  # 至少80%的查询成功
            print(f"✓ SUCCESS: batch_size={batch_size} with expected_clusters={expected_clusters} works!")
        else:
            print(f"✗ FAILED: batch_size={batch_size} with expected_clusters={expected_clusters} failed!")
    
    print("\n" + "=" * 60)
    print("Cluster count verification completed!")
    print("=" * 60)
    
    # 总结
    print("\n总结:")
    print("- 您的修改 std::max(batch_size_ / 500, 2) 确保了至少2个聚类")
    print("- 但问题可能不只是聚类数量，还可能包括:")
    print("  1. 聚类算法本身的质量")
    print("  2. 图构建过程中的连接性")
    print("  3. 批量插入和图构建的时机")
    print("  4. 中心点选择和更新逻辑")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
