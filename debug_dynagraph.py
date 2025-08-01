#!/usr/bin/env python3
"""
DynaGraph Debug Script - 测试批处理问题
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append('DynaGraph/bindings/build')
sys.path.append('big-ann-benchmarks')

try:
    from neurips23.streaming.dynagraph.dynagraph import DynaGraph
    
    print("=== DynaGraph Batch Processing Debug ===")
    
    # 测试小数据集 (< batch_size)
    print("\n--- Test 1: Small dataset (< 1000 points) ---")
    dg1 = DynaGraph('euclidean', {'batch_size': 1000})
    dg1.setup(np.float32, 10000, 128)
    
    X1 = np.random.random((500, 128)).astype(np.float32)
    ids1 = np.arange(500, dtype=np.uint64)
    
    print(f"Before insert: is_built = {dg1.is_built}")
    dg1.insert(X1, ids1)
    print(f"After insert: is_built = {dg1.is_built}")
    
    # 查询测试
    query1 = X1[0:1]  # 使用第一个点作为查询
    result1 = dg1.query(query1, 10)
    print(f"Query result shape: {result1.shape}")
    print(f"Query result[0]: {result1[0]}")
    print(f"Non-zero results: {np.count_nonzero(result1[0])}")
    
    # 测试大数据集分批插入
    print("\n--- Test 2: Large dataset (分批插入) ---")
    dg2 = DynaGraph('euclidean', {'batch_size': 1000})
    dg2.setup(np.float32, 10000, 128)
    
    # 第一批：800个点 (< batch_size)
    X2a = np.random.random((800, 128)).astype(np.float32)
    ids2a = np.arange(800, dtype=np.uint64)
    
    print(f"Before first batch: is_built = {dg2.is_built}")
    dg2.insert(X2a, ids2a)
    print(f"After first batch: is_built = {dg2.is_built}")
    
    # 第二批：500个点 (会触发 insert_concurrent)
    X2b = np.random.random((500, 128)).astype(np.float32)
    ids2b = np.arange(800, 1300, dtype=np.uint64)
    
    print(f"Before second batch: is_built = {dg2.is_built}")
    dg2.insert(X2b, ids2b)
    print(f"After second batch: is_built = {dg2.is_built}")
    
    # 查询测试
    query2 = X2a[0:1]  # 使用第一批中的点作为查询
    result2 = dg2.query(query2, 10)
    print(f"Query result shape: {result2.shape}")
    print(f"Query result[0]: {result2[0]}")
    print(f"Non-zero results: {np.count_nonzero(result2[0])}")
    
    # 测试大数据集一次性插入
    print("\n--- Test 3: Large dataset (一次性插入) ---")
    dg3 = DynaGraph('euclidean', {'batch_size': 1000})
    dg3.setup(np.float32, 10000, 128)
    
    X3 = np.random.random((1500, 128)).astype(np.float32)
    ids3 = np.arange(1500, dtype=np.uint64)
    
    print(f"Before insert: is_built = {dg3.is_built}")
    dg3.insert(X3, ids3)
    print(f"After insert: is_built = {dg3.is_built}")
    
    # 查询测试
    query3 = X3[0:1]  # 使用插入的点作为查询
    result3 = dg3.query(query3, 10)
    print(f"Query result shape: {result3.shape}")
    print(f"Query result[0]: {result3[0]}")
    print(f"Non-zero results: {np.count_nonzero(result3[0])}")
    
    print("\n=== Analysis ===")
    print("Test 1 (Small): Non-zero results =", np.count_nonzero(result1[0]))
    print("Test 2 (Batch): Non-zero results =", np.count_nonzero(result2[0]))
    print("Test 3 (Large): Non-zero results =", np.count_nonzero(result3[0]))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
