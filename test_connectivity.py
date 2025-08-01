#!/usr/bin/env python3
"""
分析DynaGraph批量插入过程中的图连通性
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append('DynaGraph/bindings/build')
sys.path.append('big-ann-benchmarks')

print("=" * 60)
print("DynaGraph Batch Insert Analysis")
print("=" * 60)

try:
    from neurips23.streaming.dynagraph.dynagraph import DynaGraph
    print("✓ DynaGraph streaming adapter imported successfully")
    
    # 分析不同batch_size下的批量插入行为
    n_points = 3000
    n_dim = 128
    
    batch_sizes = [500, 1000, 2000]
    
    for batch_size in batch_sizes:
        print(f"\n--- Analyzing batch_size={batch_size} ---")
        
        # 计算预期的批量插入次数
        expected_batches = n_points // batch_size
        remaining_points = n_points % batch_size
        
        print(f"Expected batch insertions: {expected_batches}")
        print(f"Remaining points in buffer: {remaining_points}")
        print(f"Expected clusters per batch: {max(batch_size // 500, 2)}")
        
        # 创建DynaGraph实例
        dg = DynaGraph('euclidean', {
            'alpha': 1.2, 
            'coef_L': 100, 
            'coef_R': 64,
            'batch_size': batch_size
        })
        
        max_pts = 10000
        dg.setup(np.float32, max_pts, n_dim)
        
        # 生成测试数据
        X = np.random.random((n_points, n_dim)).astype(np.float32)
        ids = np.arange(n_points, dtype=np.uint64)
        
        print(f"Inserting {n_points} points...")
        dg.insert(X, ids)
        print("Insert completed")
        
        # 测试查询连通性
        print("Testing query connectivity...")
        
        # 从不同的点开始查询，看看能否找到结果
        test_queries = []
        # 选择来自不同批次的点作为查询
        for batch_idx in range(min(expected_batches, 3)):  # 最多测试3个批次
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_points)
            if start_idx < n_points:
                # 从这个批次中选择几个点
                batch_queries = np.random.choice(range(start_idx, end_idx), 
                                                 min(3, end_idx - start_idx), 
                                                 replace=False)
                for query_idx in batch_queries:
                    test_queries.append(query_idx)
        
        print(f"Testing with {len(test_queries)} queries from different batches...")
        
        queries_matrix = X[test_queries]
        results = dg.query(queries_matrix, 10)
        
        # 分析结果
        successful_queries = 0
        for i, result in enumerate(results):
            non_zero_count = np.count_nonzero(result)
            query_batch = test_queries[i] // batch_size
            print(f"  Query {i} (from batch {query_batch}): {non_zero_count} results")
            if non_zero_count > 0:
                successful_queries += 1
        
        success_rate = successful_queries / len(test_queries) if test_queries else 0
        print(f"Success rate: {successful_queries}/{len(test_queries)} ({success_rate:.2%})")
        
        # 分析问题
        if success_rate < 0.5:
            print("❌ CONNECTIVITY ISSUE DETECTED:")
            if expected_batches > 1:
                print(f"   - Multiple batch insertions ({expected_batches}) may cause graph fragmentation")
                print(f"   - Each batch creates separate connected components")
                print(f"   - Query may not be able to traverse between components")
            if remaining_points > 0:
                print(f"   - {remaining_points} points remain in batch buffer, not in graph")
                print(f"   - These points rely on batch_index for queries")
        else:
            print("✅ GOOD CONNECTIVITY:")
            if expected_batches <= 1:
                print(f"   - Single batch insertion creates well-connected graph")
            if remaining_points > 0:
                print(f"   - Batch buffer ({remaining_points} points) provides backup via batch_index")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("问题根源:")
    print("1. 小的batch_size导致多次批量插入")
    print("2. 每次批量插入创建独立的图组件")
    print("3. 不同批次之间的图组件缺乏连接")
    print("4. 查询时无法在断开的组件间导航")
    print("\n解决方案:")
    print("1. 增大batch_size减少批量插入次数")
    print("2. 改进批量插入逻辑，确保新旧图组件连接") 
    print("3. 改进中心点选择和更新策略")
    print("4. 加强图的连通性检查和修复")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
