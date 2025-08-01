#!/usr/bin/env python3
"""
调试存储管理问题的详细测试脚本
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append('DynaGraph/bindings/build')
sys.path.append('big-ann-benchmarks')

print("=" * 60)
print("存储管理调试测试")
print("=" * 60)

try:
    # 直接使用底层绑定进行测试
    import dynagraph
    print("✓ 直接导入dynagraph绑定成功")
    
    # 测试不同的batch_size
    batch_sizes = [500, 2000]
    
    for batch_size in batch_sizes:
        print(f"\n--- 测试 batch_size = {batch_size} ---")
        
        # 创建索引
        index = dynagraph.Index()
        print("✓ 索引创建成功")
        
        # 设置参数
        max_pts = 10000
        dim = 128
        alpha = 1.2
        coef_L = 80
        coef_R = 15
        
        index.setup(max_pts, dim, alpha, coef_L, coef_R, batch_size)
        print(f"✓ 索引设置完成: batch_size={batch_size}")
        
        # 准备测试数据
        n_points = 3000
        X = np.random.random((n_points, dim)).astype(np.float32)
        ids = np.arange(n_points, dtype=np.uint64)
        
        print(f"✓ 准备 {n_points} 个点的测试数据")
        
        # 使用build方法 - 这应该同时处理存储和索引
        index.build(X, n_points, ids.tolist())
        print(f"✓ Build完成")
        
        # 测试查询
        print("--- 测试查询 ---")
        test_query = X[0]  # 使用第一个点作为查询
        k = 10
        
        try:
            tags, distances = index.query(test_query, k)
            print(f"✓ 查询成功返回 {len(tags)} 个结果")
            print(f"  标签: {tags[:5]}...")  # 显示前5个
            print(f"  距离: {distances[:5]}...")  # 显示前5个距离
            
            # 检查是否有有效结果
            valid_results = len([t for t in tags if t != 0])
            print(f"✓ 有效结果数量: {valid_results}")
            
            if valid_results > 0:
                print(f"✅ SUCCESS: batch_size={batch_size} 查询正常!")
            else:
                print(f"❌ FAILED: batch_size={batch_size} 无有效结果")
                
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("现在测试增量插入...")
    print("=" * 60)
    
    # 测试增量插入模式
    index = dynagraph.Index()
    index.setup(10000, 128, 1.2, 40, 15, 500)  # 使用小batch_size
    
    # 先用build插入一部分数据
    n_initial = 1000
    X_initial = np.random.random((n_initial, 128)).astype(np.float32)
    ids_initial = np.arange(n_initial, dtype=np.uint64)
    
    index.build(X_initial, n_initial, ids_initial.tolist())
    print(f"✓ 初始build {n_initial} 个点")
    
    # 测试查询初始数据
    test_query = X_initial[0]
    try:
        tags, distances = index.query(test_query, 5)
        print(f"✓ 初始数据查询成功: {len(tags)} 个结果")
        
        # 现在增量插入更多数据
        n_incremental = 500
        X_incremental = np.random.random((n_incremental, 128)).astype(np.float32)
        ids_incremental = np.arange(n_initial, n_initial + n_incremental, dtype=np.uint64)
        
        print(f"开始增量插入 {n_incremental} 个点...")
        success_count = 0
        for i in range(n_incremental):
            point = X_incremental[i]
            point_id = ids_incremental[i]
            try:
                success = index.insert_point(point, point_id)
                if success:
                    success_count += 1
            except Exception as e:
                print(f"插入点 {point_id} 失败: {e}")
        
        print(f"✓ 增量插入完成: {success_count}/{n_incremental} 成功")
        
        # 测试查询增量插入的数据
        if n_incremental > 0:
            test_incremental_query = X_incremental[0]
            try:
                tags, distances = index.query(test_incremental_query, 5)
                print(f"✓ 增量数据查询: {len(tags)} 个结果")
                
                # 检查是否包含增量插入的点
                incremental_found = any(tag >= n_initial for tag in tags if tag != 0)
                print(f"增量插入的点是否在结果中: {incremental_found}")
                
            except Exception as e:
                print(f"❌ 增量数据查询失败: {e}")
        
    except Exception as e:
        print(f"❌ 初始查询失败: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("调试测试完成!")
print("=" * 60)
