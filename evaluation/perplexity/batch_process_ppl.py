"""
批量处理JSON文件中的数据，计算困惑度

使用方式:
    # 使用掩码模型
    python batch_process_ppl.py input.json output.json --model_type masked
    
    # 使用自回归模型
    python batch_process_ppl.py input.json output.json --model_type autoregressive --model_name Qwen/Qwen2.5-7B
"""

import json
import argparse
import sys
import os
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from ppl_model import PerplexityCalculator


def load_json(file_path):
    """加载JSON文件"""
    print(f"加载数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：列表或字典
    if isinstance(data, dict):
        # 如果是字典，尝试提取列表
        if 'data' in data:
            data = data['data']
        elif 'items' in data:
            data = data['items']
        else:
            # 单个数据项
            data = [data]
    
    print(f"加载了 {len(data)} 个数据项")
    return data


def save_json(data, file_path):
    """保存JSON文件"""
    print(f"保存结果到: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("保存完成")


def process_file(input_file, output_file, model_name, model_type, fp16, 
                use_data_parallel, batch_size):
    """处理文件"""
    # 加载数据
    data_items = load_json(input_file)
    
    # 初始化计算器
    print(f"\n初始化模型...")
    print(f"  模型名称: {model_name}")
    print(f"  模型类型: {model_type}")
    print(f"  半精度: {fp16}")
    print(f"  多卡并行: {use_data_parallel}")
    
    calculator = PerplexityCalculator(
        model_name=model_name,
        model_type=model_type,
        fp16=fp16,
        use_data_parallel=use_data_parallel
    )
    
    # 批量处理 - 使用优化的跨词批处理
    print(f"\n开始处理数据...")
    print(f"  处理模式: 跨词批处理（高效）")
    print(f"  批大小: {batch_size}")
    
    # 使用优化的批处理方法
    results = calculator.process_batch_optimized(data_items, batch_size=batch_size)
    
    # 保存结果
    save_json(results, output_file)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"总数据项: {len(results)}")
    
    valid_results = [r for r in results if not 'error' in r and r['mean_perplexity'] != float('inf')]
    print(f"成功处理: {len(valid_results)}")
    print(f"失败: {len(results) - len(valid_results)}")
    
    if valid_results:
        perplexities = [r['mean_perplexity'] for r in valid_results]
        import numpy as np
        print(f"\n困惑度统计:")
        print(f"  最小值: {np.min(perplexities):.4f}")
        print(f"  最大值: {np.max(perplexities):.4f}")
        print(f"  平均值: {np.mean(perplexities):.4f}")
        print(f"  中位数: {np.median(perplexities):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='批量计算困惑度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认掩码模型
  python batch_process_ppl.py input.json output.json
  
  # 使用自回归模型
  python batch_process_ppl.py input.json output.json --model_type autoregressive --model_name Qwen/Qwen2.5-7B
  
  # 启用半精度和多GPU
  python batch_process_ppl.py input.json output.json --fp16 --use_data_parallel
        """
    )
    
    parser.add_argument('--input_file', type=str, help='输入JSON文件路径')
    parser.add_argument('--output_file', type=str, help='输出JSON文件路径')
    
    parser.add_argument('--model_name', type=str, 
                       default='answerdotai/ModernBERT-large',
                       help='模型名称或路径 (默认: answerdotai/ModernBERT-large)')
    
    parser.add_argument('--model_type', type=str, 
                       choices=['masked', 'autoregressive'],
                       default='masked',
                       help='模型类型: masked 或 autoregressive (默认: masked)')
    
    parser.add_argument('--fp16', action='store_true',
                       help='使用半精度推理（fp16）')
    
    parser.add_argument('--use_data_parallel', action='store_true',
                       help='启用单机多卡并行（DataParallel）')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小，推荐16-64（默认: 32）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 处理文件
    process_file(
        args.input_file,
        args.output_file,
        args.model_name,
        args.model_type,
        args.fp16,
        args.use_data_parallel,
        args.batch_size
    )


if __name__ == "__main__":
    main()
