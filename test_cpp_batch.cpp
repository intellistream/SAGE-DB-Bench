#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <memory>
#include "index/all_index.h"
#include "storage/storage_manager.h"
#include "compute_engine/compute_engine.h"

void test_batch_size(int batch_size) {
    std::cout << "\n=== Testing batch_size = " << batch_size << " ===" << std::endl;
    
    // 创建配置
    candy::DynaGraphConfig config;
    config.alpha = 1.00f;
    config.coef_L = 50;
    config.coef_R = 15;
    config.batch_size = batch_size;
    config.batch_cluster_cnt = std::max(batch_size / 500, 2);  // 和用户修改保持一致
    
    // 创建组件
    auto compute_engine = std::make_shared<candy::ComputeEngine>();
    auto storage_manager = std::make_shared<candy::StorageManager>();
    storage_manager->engine_ = compute_engine;
    
    // 创建索引
    auto index = std::make_unique<candy::DynaGraph>(config);
    index->storage_manager_ = storage_manager;
    index->dimension_ = 128;
    
    // 生成测试数据
    const int n_points = 10000;
    const int dimension = 128;
    std::vector<std::vector<float>> data(n_points, std::vector<float>(dimension));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < dimension; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    std::cout << "Generated " << n_points << " points of dimension " << dimension << std::endl;
    
    // 插入数据到存储管理器和索引
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int successful_inserts = 0;
    int failed_inserts = 0;
    
    for (int i = 0; i < n_points; ++i) {
        uint64_t uid = i;
        
        // 创建VectorData
        candy::VectorData vector_data(dimension, candy::DataType::Float32);
        float* data_ptr = reinterpret_cast<float*>(vector_data.data_.get());
        std::memcpy(data_ptr, data[i].data(), dimension * sizeof(float));
        
        // 创建VectorRecord
        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        auto record = std::make_unique<candy::VectorRecord>(uid, timestamp, std::move(vector_data));
        
        // 存储到StorageManager
        storage_manager->insert(record);
        
        // 插入到索引 - 添加详细调试信息
        if (i < 10 || i % 1000 == 0) {  // 只对前10个和每1000个点打印详细信息
            std::cout << "Inserting point " << uid << "..." << std::flush;
        }
        bool success = index->insert(uid);
        
        if (success) {
            successful_inserts++;
            if (i < 10 || i % 1000 == 0) {
                std::cout << " SUCCESS" << std::endl;
            }
        } else {
            failed_inserts++;
            if (i < 10 || i % 1000 == 0) {
                std::cout << " FAILED" << std::endl;
            }
        }
        
        // 每1000个点打印一次进度
        if ((i + 1) % 1000 == 0) {
            std::cout << "Progress: " << (i + 1) << "/" << n_points << " points inserted" << std::endl;
        }
    }
    
    std::cout << "Insert statistics: " << successful_inserts << " successful, " << failed_inserts << " failed" << std::endl;
    
    // 强制触发最后的batch_insert（如果还有未处理的batch）
    std::cout << "Forcing final batch insert..." << std::endl;
    // 注意：DynaGraph可能需要手动触发最后的批处理
    // 我们可以尝试插入一个额外的点来触发batch_insert
    if (n_points % batch_size != 0) {
        std::cout << "There might be pending batch items, inserting trigger point..." << std::endl;
        for (int trigger = 0; trigger < batch_size; ++trigger) {
            uint64_t trigger_uid = n_points + trigger;
            
            // 创建trigger point
            candy::VectorData trigger_data(dimension, candy::DataType::Float32);
            float* trigger_ptr = reinterpret_cast<float*>(trigger_data.data_.get());
            for (int j = 0; j < dimension; ++j) {
                trigger_ptr[j] = 0.5f;  // 简单的trigger数据
            }
            
            auto trigger_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            auto trigger_record = std::make_unique<candy::VectorRecord>(trigger_uid, trigger_timestamp, std::move(trigger_data));
            storage_manager->insert(trigger_record);
            
            bool trigger_success = index->insert(trigger_uid);
            if (!trigger_success) {
                std::cout << "Trigger insert failed" << std::endl;
                break;
            }
            
            // 如果成功触发了batch_insert，就停止
            if ((n_points + trigger + 1) % batch_size == 0) {
                std::cout << "Batch insert triggered after " << (trigger + 1) << " trigger points" << std::endl;
                break;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Insert completed in " << duration.count() << "ms" << std::endl;
    
    // 测试查询
    std::cout << "Testing queries..." << std::endl;
    int successful_queries = 0;
    const int num_test_queries = 5;
    
    for (int q = 0; q < num_test_queries; ++q) {
        int test_idx = q * (n_points / num_test_queries);  // 分散选择测试点
        
        // 创建查询向量
        candy::VectorData query_data(dimension, candy::DataType::Float32);
        float* query_ptr = reinterpret_cast<float*>(query_data.data_.get());
        std::memcpy(query_ptr, data[test_idx].data(), dimension * sizeof(float));
        
        auto query_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        static int ppoint_id = 999999;  // 固定的查询点 ID
        auto query_record = std::make_unique<candy::VectorRecord>(
            ppoint_id --, query_timestamp, std::move(query_data));
        
        //storage_manager -> insert(query_record);

        try {
            auto results = index->query(query_record, 10);
            
            if (!results.empty()) {
                successful_queries++;
                std::cout << "Query " << q << " returned " << results.size() << " results";
                if (results.size() > 0) {
                    std::cout << " (first result: " << results[0] << ")";
                }
                std::cout << std::endl;
            } else {
                std::cout << "Query " << q << " returned 0 results" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Query " << q << " failed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Successful queries: " << successful_queries << "/" << num_test_queries << std::endl;
    
    if (successful_queries > 0) {
        std::cout << "✅ SUCCESS: batch_size=" << batch_size << " works in C++!" << std::endl;
    } else {
        std::cout << "❌ FAILED: batch_size=" << batch_size << " failed in C++!" << std::endl;
    }
}

int main() {
    std::cout << "=== DynaGraph C++ Batch Size Test ===" << std::endl;
    
    // 测试不同的batch_size
    std::vector<int> batch_sizes = {500, 1000, 2000};
    
    for (int batch_size : batch_sizes) {
        test_batch_size(batch_size);
    }
    
    std::cout << "\n=== C++ Test Completed ===" << std::endl;
    return 0;
}
