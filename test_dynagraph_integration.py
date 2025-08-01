#!/usr/bin/env python3

import numpy as np
import sys
import os

def test_streaming_integration():
    """测试DynaGraph与streaming benchmark的集成"""
    
    print("=== Testing DynaGraph Streaming Integration ===")
    
    # 模拟streaming benchmark的调用模式
    try:
        # 导入我们的streaming适配器
        sys.path.append('/Users/rprp/Desktop/Github/SAGE-DB-Bench/big-ann-benchmarks')
        from neurips23.streaming.dynagraph.dynagraph import DynaGraph
        
        print("✓ DynaGraph streaming adapter imported successfully")
        
        # 创建索引实例
        metric = "euclidean"
        index_params = {
            "alpha": 1.2,
            "coef_L": 100, 
            "coef_R": 64,
            "batch_size": 500,
            "insert_thread_count": 4,
            "search_thread_count": 4
        }
        
        dynagraph_index = DynaGraph(metric, index_params)
        print("✓ DynaGraph index created")
        
        # 设置索引
        dtype = "float32"
        max_pts = 10000
        ndim = 128
        
        dynagraph_index.setup(dtype, max_pts, ndim)
        print("✓ Index setup completed")
        
        # 模拟streaming benchmark的数据流
        print("\n--- Simulating streaming operations ---")
        
        # 初始构建
        n_initial = 1000
        X_initial = np.random.random((n_initial, ndim)).astype(np.float32)
        ids_initial = np.arange(n_initial, dtype=np.uint32)
        
        dynagraph_index.insert(X_initial, ids_initial)
        print(f"✓ Initial insert: {n_initial} points")
        
        # 查询
        n_queries = 10
        k = 10
        X_queries = np.random.random((n_queries, ndim)).astype(np.float32)
        
        dynagraph_index.query(X_queries, k)
        print(f"✓ Query: {n_queries} queries, k={k}")
        
        # 增量插入
        n_incremental = 100
        X_incremental = np.random.random((n_incremental, ndim)).astype(np.float32)
        ids_incremental = np.arange(n_initial, n_initial + n_incremental, dtype=np.uint32)
        
        dynagraph_index.insert(X_incremental, ids_incremental)
        print(f"✓ Incremental insert: {n_incremental} points")
        
        # 删除
        n_delete = 50
        ids_to_delete = np.arange(0, n_delete, dtype=np.uint32)
        
        dynagraph_index.delete(ids_to_delete)
        print(f"✓ Delete: {n_delete} points")
        
        # 最终查询
        dynagraph_index.query(X_queries, k)
        print(f"✓ Final query: {n_queries} queries, k={k}")
        
        print("\n=== Streaming integration test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_integration()
    sys.exit(0 if success else 1)
