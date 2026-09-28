# 分布式推理文档

## 概述

使用训练好的GlossBERT模型对大规模上下文-释义对进行推理，筛选出匹配的样本。支持单机多卡和多机多卡部署。

## 功能特点

- ✅ 基于torchrun的分布式推理
- ✅ 支持单机多卡和多机多卡
- ✅ 自动数据分片和结果合并
- ✅ 内存高效的流式处理
- ✅ 可配置的置信度阈值
- ✅ 处理20M+样本的能力

## 环境要求

```bash
# Python包
torch>=1.12.0
transformers>=4.20.0
tqdm

# 硬件
- GPU: 推荐多卡 (4-8 GPUs per node)
- 显存: 每个GPU至少8GB (batch_size=32时)
- 内存: 推荐32GB+
```

## 使用方法

### 1. 准备数据

确保已经运行了 `corpus_filter.py`，生成了以下文件：
```
corpus_filter_output/
├── existing_words_contexts.jsonl  (20M+ 行)
├── positive_words_contexts.jsonl
└── negative_words_contexts.jsonl
```

每行格式：
```json
{
    "word": "轻贱",
    "sense": "低贱；下贱。",
    "context": "...上下文...",
    "word_start": 10,
    "word_end": 12
}
```

### 2. 准备模型

确保有训练好的模型检查点：
```
checkpoints/best_model/
└── pytorch_model.bin
```

### 3. 单机多卡推理

**Linux/Mac:**
```bash
# 修改 run_inference_single_node.sh 中的参数
chmod +x run_inference_single_node.sh
./run_inference_single_node.sh
```

**Windows:**
```bash
# 修改 run_inference.bat 中的参数
run_inference.bat
```

**直接命令:**
```bash
torchrun --nproc_per_node=4 \
    inference_distributed.py \
    --input_dir ./corpus_filter_output \
    --model_path ./checkpoints/best_model \
    --output_dir ./filtered_results \
    --batch_size 64 \
    --threshold 0.5
```

### 4. 多机多卡推理

#### 环境准备
1. 确保所有节点可以互相访问
2. 所有节点安装相同的环境
3. 所有节点可以访问相同的数据和模型路径（通过NFS或拷贝）

#### 运行步骤

**主节点 (Node 0):**
```bash
# 修改 run_inference_multi_node_0.sh
# 设置 MASTER_ADDR 为主节点的IP
chmod +x run_inference_multi_node_0.sh
./run_inference_multi_node_0.sh
```

**从节点 (Node 1, 2, ...):**
```bash
# 修改 run_inference_multi_node_1.sh
# NODE_RANK 设置为 1, 2, 3...
# MASTER_ADDR 保持与主节点相同
chmod +x run_inference_multi_node_1.sh
./run_inference_multi_node_1.sh
```

## 参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--input_dir` | corpus_filter输出目录 | 必需 | - |
| `--model_path` | 模型检查点路径 | 必需 | - |
| `--output_dir` | 输出目录 | 必需 | - |
| `--batch_size` | 每GPU批次大小 | 32 | 32-128 |
| `--max_length` | 最大序列长度 | 512 | 512 |
| `--threshold` | 置信度阈值 | 0.5 | 0.5-0.7 |
| `--num_workers` | DataLoader工作进程 | 4 | 4-8 |

## 输出格式

生成三个过滤后的文件：
```
filtered_results/
├── existing_words_filtered.jsonl
├── positive_words_filtered.jsonl
└── negative_words_filtered.jsonl
```

每行格式：
```json
{
    "word": "轻贱",
    "context": "...上下文...",
    "sense": "低贱；下贱。",
    "word_start": 10,
    "word_end": 12,
    "probability": 0.87  // 模型预测的匹配概率
}
```

## 性能优化

### 批次大小调优

根据GPU显存调整 `batch_size`：

| GPU显存 | 推荐batch_size | 预期速度 (样本/秒/GPU) |
|---------|---------------|----------------------|
| 8GB | 16-32 | ~100-200 |
| 11GB | 32-64 | ~200-400 |
| 16GB | 64-128 | ~400-800 |
| 24GB+ | 128-256 | ~800-1500 |

### 多机配置建议

**2机8卡配置 (处理20M样本):**
- 每机4卡 × 2机 = 8卡
- batch_size=64
- 预计速度: ~2000-3000 样本/秒
- 预计时间: ~2-3小时

**4机16卡配置 (处理20M样本):**
- 每机4卡 × 4机 = 16卡
- batch_size=64
- 预计速度: ~4000-6000 样本/秒
- 预计时间: ~1-1.5小时

### 数据加载优化

```bash
# 增加workers提升IO速度
--num_workers 8

# 如果内存充足，可以增加预取
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 阈值选择

`--threshold` 参数控制样本过滤的严格程度：

| 阈值 | 含义 | 适用场景 |
|------|------|---------|
| 0.3 | 宽松 | 召回率优先，保留更多样本 |
| 0.5 | 平衡 | 准确率和召回率平衡 |
| 0.7 | 严格 | 准确率优先，只保留高置信度样本 |

建议：
1. 先用 `threshold=0.5` 运行
2. 查看结果分布
3. 根据实际需求调整

## 故障排查

### 1. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
- 减小 `--batch_size`
- 减小 `--max_length`
- 减少 `--num_workers`

### 2. 多机连接失败
```
socket timeout
```
**解决方案:**
- 检查防火墙设置
- 确认 `MASTER_ADDR` 和 `MASTER_PORT` 正确
- 确保所有节点网络互通

### 3. 数据路径问题
```
FileNotFoundError
```
**解决方案:**
- 使用绝对路径
- 确保所有节点都能访问数据（NFS或本地拷贝）

### 4. 进程挂起
**解决方案:**
- 检查是否所有节点都启动了
- 确认 `--nnodes` 和实际启动的节点数一致

## 监控和日志

### 查看GPU使用情况
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用
nvitop
```

### 查看进程
```bash
# 查看Python进程
ps aux | grep inference_distributed

# 查看网络连接
netstat -an | grep 29500
```

### 估算剩余时间
脚本会显示进度条和处理速度，可以根据以下公式估算：
```
剩余时间 = 剩余样本数 / (速度 × GPU数)
```

## 示例工作流

```bash
# 1. 提取上下文 (corpus_filter.py)
python corpus_filter.py \
    --word_list word_list.json \
    --corpus large_corpus.jsonl \
    --output_dir corpus_filter_output

# 2. 分布式推理过滤 (本脚本)
torchrun --nproc_per_node=4 \
    inference_distributed.py \
    --input_dir corpus_filter_output \
    --model_path checkpoints/best_model \
    --output_dir filtered_results \
    --batch_size 64 \
    --threshold 0.5

# 3. 查看结果统计
wc -l filtered_results/*.jsonl
```

## 注意事项

1. **数据一致性**: 确保所有节点使用相同的输入数据
2. **模型一致性**: 确保所有节点使用相同的模型检查点
3. **同步启动**: 多机推理时需要在所有节点几乎同时启动脚本
4. **磁盘空间**: 确保输出目录有足够空间存储结果
5. **中断恢复**: 当前版本不支持断点续传，中断后需要重新运行

## 许可证

内部使用
