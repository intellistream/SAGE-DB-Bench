# DynaGraph Integration with SAGE-DB-Bench

DynaGraph已成功集成到SAGE-DB-Bench基准测试框架中，支持streaming和congestion两种评测模式。

## 项目结构

```
SAGE-DB-Bench/
├── DynaGraph/                          # DynaGraph主项目
│   ├── bindings/                       # Python绑定层
│   │   ├── dynagraph_wrapper.h         # C++包装器头文件
│   │   ├── dynagraph_wrapper.cpp       # C++包装器实现
│   │   ├── pybind_module.cpp           # pybind11绑定代码
│   │   ├── CMakeLists.txt              # 绑定编译配置
│   │   └── test_bindings.py            # 绑定测试脚本
│   └── build_bindings.sh               # 构建脚本
├── big-ann-benchmarks/
│   └── neurips23/
│       ├── streaming/
│       │   └── dynagraph/              # Streaming track适配
│       │       ├── dynagraph.py        # Python适配器
│       │       ├── config.yaml         # 算法配置
│       │       └── Dockerfile          # Docker构建文件
│       └── congestion/
│           └── dynagraph/              # Congestion track适配
│               ├── dynagraph.py        # Python适配器
│               └── config.yaml         # 算法配置
└── test_dynagraph_integration.py       # 集成测试脚本
```

## 数据传递流程

### 1. 数据源 → Benchmark框架
```python
# benchmark框架从数据集文件读取数据
ds = DATASETS[dataset]()
X = ds.get_data_in_range(start, end)  # numpy array (float32)
ids = np.arange(start, end, dtype=np.uint32)
```

### 2. Benchmark框架 → Python适配层
```python
# neurips23/streaming/run.py
algo.insert(X, ids)  # X: numpy array, ids: numpy array
```

### 3. Python适配层 → pybind11绑定层
```python
# neurips23/streaming/dynagraph/dynagraph.py
def insert(self, X, ids):
    X = X.astype(np.float32)         # 确保数据类型
    ids = ids.astype(np.uint64)      # DynaGraph使用uint64_t
    self.index.insert_concurrent(X, ids, self.insert_thread_count)
```

### 4. pybind11绑定层 → C++包装器
```cpp
// bindings/pybind_module.cpp
.def("insert_concurrent", [](DynaGraphWrapper& self,
                py::array_t<float> X,
                py::array_t<uint64_t> tags,
                int32_t thread_count) {
    // 直接传递numpy array的数据指针，零拷贝
    auto results = self.insert_points_concurrent(
        X.data(),           // float* 原始数据指针
        tags.data(),        // uint64_t* ID指针
        num_points, dim, thread_count
    );
})
```

### 5. C++包装器 → DynaGraph核心
```cpp
// bindings/dynagraph_wrapper.cpp
std::vector<bool> DynaGraphWrapper::insert_points_concurrent(...) {
    for (int64_t i = 0; i < num_points; ++i) {
        const float* point = data + i * dim;  // 指针算术，无数据拷贝
        uint64_t tag = tags[i];
        
        // 创建VectorRecord并存储到StorageManager
        auto record = createVectorRecord(tag, point, dimension_);
        storage_manager_->insert(record);
        
        // 插入到DynaGraph索引
        results[i] = index_->insert(tag);
    }
}
```

## 关键特性

### 1. 零拷贝数据传递
- 数据从Python numpy array直接通过指针传递到C++
- 使用`py::array_t<float, py::array::c_style | py::array::forcecast>`确保内存布局兼容
- `X.data()`获取原始内存指针，避免额外拷贝

### 2. DynaGraph特有的数据处理
- **批量插入机制**：DynaGraph内部使用批量插入优化性能
- **聚类策略**：支持批量插入时的聚类操作
- **增量构建**：通过insert操作逐步构建索引，而非一次性build

### 3. 多Track支持
- **Streaming Track**：直接继承`BaseStreamingANN`
- **Congestion Track**：通过包装streaming实现继承`BaseCongestionDropANN`

### 4. 参数配置
```yaml
# config.yaml示例
dynagraph:
  args: |
    [{
      "alpha": 1.2,           # 图修剪参数
      "coef_L": 100,          # 搜索候选集大小
      "coef_R": 64,           # 每个节点最大出度
      "batch_size": 2000,     # 批量插入阈值
      "insert_thread_count": 8,
      "search_thread_count": 8
    }]
```

## 构建和测试

### 1. 构建DynaGraph和Python绑定
```bash
cd /Users/rprp/Desktop/Github/SAGE-DB-Bench/DynaGraph
./build_bindings.sh
```

### 2. 测试Python绑定
```bash
cd /Users/rprp/Desktop/Github/SAGE-DB-Bench/DynaGraph/bindings
python3 test_bindings.py
```

### 3. 测试Benchmark集成
```bash
cd /Users/rprp/Desktop/Github/SAGE-DB-Bench
python3 test_dynagraph_integration.py
```

### 4. Docker构建（用于benchmark运行）
```bash
cd /Users/rprp/Desktop/Github/SAGE-DB-Bench/big-ann-benchmarks/neurips23/streaming/dynagraph
docker build -t neurips23-streaming-dynagraph .
```

## 与IP-DiskANN的对比

| 特性 | IP-DiskANN | DynaGraph |
|------|------------|-----------|
| ID类型 | uint32_t (+1偏移) | uint64_t (无偏移) |
| 数据类型 | float32 | float32 |
| 构建方式 | build() + insert() | 纯增量insert() |
| 批量操作 | 支持 | 内建批量优化 |
| 内存管理 | DiskANN存储 | StorageManager + ComputeEngine |
| 图结构 | Vamana图 | 动态图结构 |

## 使用示例

```python
# 创建DynaGraph索引
from neurips23.streaming.dynagraph.dynagraph import DynaGraph

metric = "euclidean"
index_params = {
    "alpha": 1.2,
    "coef_L": 100,
    "coef_R": 64,
    "batch_size": 2000
}

index = DynaGraph(metric, index_params)
index.setup("float32", max_pts=100000, ndim=128)

# 插入数据
X = np.random.random((1000, 128)).astype(np.float32)
ids = np.arange(1000, dtype=np.uint32)
index.insert(X, ids)

# 查询
queries = np.random.random((10, 128)).astype(np.float32)
index.query(queries, k=10)

# 删除
index.delete(np.array([0, 1, 2], dtype=np.uint32))
```

## 注意事项

1. **线程安全**：DynaGraph的并发插入需要合适的线程数配置
2. **内存使用**：批量插入会临时增加内存使用
3. **参数调优**：alpha、coef_L、coef_R参数需要根据数据集特性调整
4. **构建依赖**：需要OpenMP支持并发操作

这个集成方案实现了DynaGraph与SAGE-DB-Bench的无缝对接，支持高效的流式数据处理和查询操作。
