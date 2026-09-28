#!/bin/bash
# 多机多卡推理脚本 - Node 0 (主节点)

# 配置参数
NUM_NODES=2  # 节点数量
NUM_GPUS_PER_NODE=4  # 每个节点的GPU数量
NODE_RANK=0  # 当前节点的rank (主节点为0)
MASTER_ADDR="192.168.1.100"  # 主节点IP地址
MASTER_PORT=29500  # 主节点端口

INPUT_DIR="./corpus_filter_output"
MODEL_PATH="../checkpoints/best_model"
OUTPUT_DIR="./filtered_results"
BATCH_SIZE=64
MAX_LENGTH=512
THRESHOLD=0.5
NUM_WORKERS=4

# 运行推理
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    inference_distributed.py \
    --input_dir $INPUT_DIR \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS
