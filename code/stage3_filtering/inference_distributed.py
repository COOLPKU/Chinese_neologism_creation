"""
分布式推理脚本 - 使用DeepSpeed和torchrun进行大规模上下文-释义匹配推理
支持多机多卡部署，处理海量数据（20M+行）

使用方法:
    单机多卡:
        torchrun --nproc_per_node=4 inference_distributed.py \
            --input_dir ./corpus_filter_output \
            --model_path ./checkpoints/best_model \
            --output_dir ./filtered_results \
            --batch_size 64 \
            --threshold 0.5
    
    多机多卡 (Node 0):
        torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
            --master_addr="192.168.1.100" --master_port=29500 \
            inference_distributed.py \
            --input_dir ./corpus_filter_output \
            --model_path ./checkpoints/best_model \
            --output_dir ./filtered_results
    
    多机多卡 (Node 1):
        torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
            --master_addr="192.168.1.100" --master_port=29500 \
            inference_distributed.py \
            --input_dir ./corpus_filter_output \
            --model_path ./checkpoints/best_model \
            --output_dir ./filtered_results
"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import BertTokenizer
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List, Dict

# 添加父目录到路径以导入model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GlossBERT


class InferenceDataset(Dataset):
    """用于推理的数据集 - 使用文件偏移量索引，避免内存溢出"""
    
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 512):
        """
        Args:
            jsonl_file: corpus_filter.py输出的jsonl文件路径
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.jsonl_file = jsonl_file
        self.offsets = []
        
        print(f"Indexing data from {jsonl_file}...")
        # 建立文件偏移量索引
        with open(jsonl_file, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
        
        print(f"Indexed {len(self.offsets)} samples")
    
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, idx):
        offset = self.offsets[idx]
        
        # 每次读取打开文件，虽然有开销，但避免了多进程文件句柄问题
        # 对于大模型推理，IO通常不是瓶颈
        with open(self.jsonl_file, 'rb') as f:
            f.seek(offset)
            line_bytes = f.readline()
            try:
                line = line_bytes.decode('utf-8')
                sample = json.loads(line.strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 容错处理，返回空样本或随机样本，避免崩溃
                # 这里简单返回一个dummy数据，实际应用中可能需要更复杂的处理
                print(f"Warning: Decode error at index {idx}")
                return self.__getitem__((idx + 1) % len(self))
        
        context = sample['context']
        sense = sample['sense']
        word = sample['word']
        word_start = sample['word_start']
        word_end = sample['word_end']
        
        # 使用与训练时相同的格式：[CLS] context [SEP] sense [SEP]
        encoding = self.tokenizer.encode_plus(
            text=context,
            text_pair=sense,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'word': word,
            'context': context,
            'sense': sense,
            'word_start': word_start,
            'word_end': word_end,
            'idx': idx  # 保存原始索引用于排序
        }


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    
    # 设置更长的超时时间
    import datetime
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(hours=2)  # 设置2小时超时
    )
    
    return rank, world_size, local_rank


def collate_fn(batch):
    """自定义collate函数，处理非张量数据"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    
    # 保留元数据
    metadata = {
        'words': [item['word'] for item in batch],
        'contexts': [item['context'] for item in batch],
        'senses': [item['sense'] for item in batch],
        'word_starts': [item['word_start'] for item in batch],
        'word_ends': [item['word_end'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'metadata': metadata
    }


@torch.no_grad()
def inference(model, dataloader, device, rank, world_size, output_file: str, threshold: float = 0.5):
    """
    执行推理并将结果流式写入文件
    
    Args:
        model: GlossBERT模型
        dataloader: 数据加载器
        device: 设备
        rank: 当前进程rank
        world_size: 总进程数
        output_file: 输出文件路径
        threshold: 置信度阈值，只保留概率>=threshold的结果
    """
    model.eval()
    
    # 打开输出文件
    f_out = open(output_file, 'w', encoding='utf-8')
    
    # 只在rank 0显示进度条
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc=f"Inference (Rank {rank})")
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        metadata = batch['metadata']
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 计算概率
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # 收集满足阈值的结果
        for i in range(len(probs)):
            prob = float(probs[i])
            
            # 只保留概率高于阈值的样本
            if prob >= threshold:
                result = {
                    'word': metadata['words'][i],
                    'context': metadata['contexts'][i],
                    'sense': metadata['senses'][i],
                    'word_start': metadata['word_starts'][i],
                    'word_end': metadata['word_ends'][i],
                    'probability': prob,
                    'idx': metadata['indices'][i]  # 保存原始索引
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        if rank == 0:
            pbar.update(1)
    
    if rank == 0:
        pbar.close()
        
    f_out.close()
    return output_file


def save_results(results: List[Dict], output_file: str, rank: int, world_size: int):
    """
    保存结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
        rank: 当前进程rank
        world_size: 总进程数
    """
    # 保存当前rank的结果到临时文件
    temp_output_file = f"{output_file}.rank{rank}.tmp"
    
    # 按原始索引排序
    results.sort(key=lambda x: x['idx'])
    
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        for result in results:
            # 移除内部使用的idx字段
            result_copy = result.copy()
            result_copy.pop('idx', None)
            f.write(json.dumps(result_copy, ensure_ascii=False) + '\n')
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    # Rank 0 合并所有结果
    if rank == 0:
        print(f"Merging results from all ranks to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for r in range(world_size):
                temp_file = f"{output_file}.rank{r}.tmp"
                if os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            outf.write(line)
                    os.remove(temp_file)  # 删除临时文件
        
        print(f"Results saved to {output_file}")


def load_model(model_path: str, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型检查点路径
        device: 设备
    
    Returns:
        model: 加载好的模型
    """
    model = GlossBERT(model_name='/mnt/model-gui-agent/bert-base-chinese')
    
    # 尝试多种可能的检查点路径
    checkpoint_paths = [
        os.path.join(model_path, 'model.safetensors'),
        os.path.join(model_path, 'pytorch_model.bin'),
        os.path.join(model_path, 'model.bin'),
        model_path  # 可能直接指向.bin文件
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path) and os.path.isfile(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    
    print(f"Loading model from {checkpoint_path}")
    
    # 加载状态字典
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device='cpu')
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 处理可能的键名前缀问题（如果用了DDP训练）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉'module.'前缀
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def process_file(
    input_file: str,
    output_file: str,
    model,
    tokenizer,
    device,
    rank: int,
    world_size: int,
    batch_size: int,
    max_length: int,
    threshold: float,
    num_workers: int = 4
):
    """
    处理单个输入文件
    
    Args:
        input_file: 输入jsonl文件
        output_file: 输出jsonl文件
        model: GlossBERT模型
        tokenizer: tokenizer
        device: 设备
        rank: 当前rank
        world_size: 总进程数
        batch_size: 批次大小
        max_length: 最大序列长度
        threshold: 置信度阈值
        num_workers: DataLoader工作进程数
    """
    if not os.path.exists(input_file):
        if rank == 0:
            print(f"Input file not found: {input_file}")
        return
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Processing: {input_file}")
        print(f"Output: {output_file}")
        print(f"{'='*80}")
    
    # 创建数据集
    dataset = InferenceDataset(
        jsonl_file=input_file,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    if len(dataset) == 0:
        if rank == 0:
            print(f"No valid samples in {input_file}, skipping...")
        return
    
    # 创建分布式采样器
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False  # 推理时不打乱
        )
    else:
        sampler = None
    
    # 创建数据加载器
    # 注意：分布式推理时使用 drop_last=True 避免不同 rank 的 batch 数量不一致导致死锁
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(world_size > 1)  # 分布式时丢弃最后不完整的batch
    )
    
    # 执行推理
    # 生成临时输出文件名
    temp_output_file = f"{output_file}.rank{rank}.tmp"
    
    inference(
        model=model,
        dataloader=dataloader,
        device=device,
        rank=rank,
        world_size=world_size,
        output_file=temp_output_file,
        threshold=threshold
    )
    
    # 等待所有进程完成
    if world_size > 1:
        # 先同步确保所有进程都完成了推理
        try:
            dist.barrier()
        except Exception as e:
            print(f"Rank {rank}: barrier failed with {e}, continuing...")
    
    # 短暂等待确保文件写入完成
    import time
    time.sleep(2)
    
    # Rank 0 合并结果
    if rank == 0:
        print("Merging results from all ranks...")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for r in range(world_size):
                rank_file = f"{output_file}.rank{r}.tmp"
                if os.path.exists(rank_file):
                    with open(rank_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            f_out.write(line)
                    os.remove(rank_file)
        print(f"Results merged to {output_file}")
    
    if rank == 0:
        # 统计最终结果
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                total_filtered = sum(1 for _ in f)
            print(f"Total samples kept: {total_filtered} / {len(dataset)} "
                  f"({100*total_filtered/len(dataset):.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Distributed inference for context-sense matching using GlossBERT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出路径
    parser.add_argument('--input_dir', type=str, required=True,
                        help='目录包含corpus_filter.py的输出文件')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型检查点路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='过滤后结果的输出目录')
    
    # 模型参数
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每个GPU的批次大小')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='置信度阈值，只保留概率>=threshold的结果')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader工作进程数')
    
    # 分布式参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (自动由torchrun设置)')
    
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("=" * 80)
        print("Distributed Inference Configuration")
        print("=" * 80)
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print(f"Input Directory: {args.input_dir}")
        print(f"Model Path: {args.model_path}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Max Length: {args.max_length}")
        print(f"Threshold: {args.threshold}")
        print(f"Num Workers: {args.num_workers}")
        print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 等待rank 0创建目录
    if world_size > 1:
        dist.barrier()
    
    # 设置设备
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('/mnt/model-gui-agent/bert-base-chinese')
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 使用DistributedDataParallel包装模型
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    # 处理三种类型的文件
    file_types = ['existing_words', 'positive_words', 'negative_words']
    
    for file_type in file_types:
        input_file = os.path.join(args.input_dir, f'{file_type}_contexts.jsonl')
        output_file = str(output_dir / f'{file_type}_filtered.jsonl')
        
        process_file(
            input_file=input_file,
            output_file=output_file,
            model=model,
            tokenizer=tokenizer,
            device=device,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            max_length=args.max_length,
            threshold=args.threshold,
            num_workers=args.num_workers
        )
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("All processing completed!")
        print("=" * 80)
    
    # 清理
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
