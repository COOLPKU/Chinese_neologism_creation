"""
批量处理脚本：处理大量数据文件并计算困惑度
"""

import json
import os
import argparse
from typing import List, Dict
import logging
from ppl_model import PerplexityCalculator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perplexity_batch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    加载JSONL格式文件
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行JSON解析错误: {e}")
    return data

def load_json_file(file_path: str) -> List[Dict]:
    """
    加载JSON格式文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果加载的是单个对象而不是列表，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    return data

def save_results(results: List[Dict], output_file: str, format_type: str = 'json'):
    """
    保存结果到文件
    
    Args:
        results: 计算结果列表
        output_file: 输出文件路径
        format_type: 输出格式 ('json' 或 'jsonl')
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if format_type.lower() == 'jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {output_file}")

def validate_data_item(item: Dict) -> bool:
    """
    验证数据项是否包含必要字段
    
    Args:
        item: 数据项
        
    Returns:
        是否有效
    """
    required_fields = ['head_id', 'tail_id', 'sentences']
    
    for field in required_fields:
        if field not in item:
            logger.warning(f"数据项缺少必要字段: {field}")
            return False
    
    if not isinstance(item['sentences'], list) or len(item['sentences']) == 0:
        logger.warning("sentences字段应为非空列表")
        return False
    
    return True

def process_batch_with_checkpoint(calculator: PerplexityCalculator, 
                                 data_items: List[Dict],
                                 checkpoint_file: str = None,
                                 batch_size: int = 10,
                                 sentence_batch_size: int = 8,
                                 use_batch_mode: bool = True) -> List[Dict]:
    """
    带检查点的批量处理
    
    Args:
        calculator: 困惑度计算器
        data_items: 数据项列表
        checkpoint_file: 检查点文件路径
        batch_size: 批次大小（用于检查点保存频率）
        sentence_batch_size: 句子批处理大小（用于GPU并行）
        use_batch_mode: 是否使用批量处理模式（提高GPU利用率）
        
    Returns:
        处理结果列表
    """
    results = []
    start_idx = 0
    
    # 如果存在检查点文件，加载已处理的结果
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                start_idx = checkpoint_data.get('processed_count', 0)
            logger.info(f"从检查点恢复，已处理 {start_idx} 个数据项")
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
    
    total_items = len(data_items)
    
    for i in range(start_idx, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_items = data_items[i:batch_end]
        
        logger.info(f"处理批次 {i+1}-{batch_end}/{total_items}")
        
        for j, item in enumerate(batch_items):
            current_idx = i + j + 1
            logger.info(f"处理第 {current_idx}/{total_items} 个数据项")
            
            try:
                # 根据模式选择处理方法
                if use_batch_mode:
                    result = calculator.calculate_sentence_perplexities_batch(
                        item, sentence_batch_size=sentence_batch_size
                    )
                else:
                    result = calculator.calculate_sentence_perplexities(item)
                
                results.append(result)
                
                # 每处理10个数据项保存一次检查点
                if current_idx % 10 == 0 and checkpoint_file:
                    checkpoint_data = {
                        'processed_count': current_idx,
                        'results': results
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"检查点已保存，已处理 {current_idx} 个数据项")
                
            except Exception as e:
                logger.error(f"处理第 {current_idx} 个数据项时出错: {e}")
                # 添加错误结果，保持原有数据结构
                error_result = item.copy()  # 复制原有数据
                error_result["target_word"] = "ERROR"
                error_result["mean_perplexity"] = float('inf')
                error_result["sentence_perplexities"] = [float('inf')] * len(item.get("sentences", []))
                error_result["error"] = str(e)
                results.append(error_result)
    
    # 删除检查点文件
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("检查点文件已删除")
    
    return results

# def generate_summary_report(results: List[Dict], output_file: str = None):
#     """
#     生成汇总报告
    
#     Args:
#         results: 计算结果列表
#         output_file: 报告输出文件路径
#     """
#     total_items = len(results)
#     valid_results = [r for r in results if r.get('mean_perplexity', float('inf')) != float('inf')]
#     error_count = total_items - len(valid_results)
    
#     if valid_results:
#         perplexities = [r['mean_perplexity'] for r in valid_results]
#         mean_ppl = sum(perplexities) / len(perplexities)
#         min_ppl = min(perplexities)
#         max_ppl = max(perplexities)
        
#         # 按困惑度排序
#         sorted_results = sorted(valid_results, key=lambda x: x['mean_perplexity'])
        
#         report = {
#             "summary": {
#                 "total_items": total_items,
#                 "valid_items": len(valid_results),
#                 "error_items": error_count,
#                 "overall_mean_perplexity": mean_ppl,
#                 "min_perplexity": min_ppl,
#                 "max_perplexity": max_ppl
#             },
#             "top_10_lowest_perplexity": sorted_results[:10],
#             "top_10_highest_perplexity": sorted_results[-10:]
#         }
#     else:
#         report = {
#             "summary": {
#                 "total_items": total_items,
#                 "valid_items": 0,
#                 "error_items": error_count,
#                 "message": "没有有效的困惑度计算结果"
#             }
#         }
    
#     # 输出到控制台
#     print("\n" + "="*60)
#     print("困惑度计算汇总报告")
#     print("="*60)
#     print(f"总数据项数: {total_items}")
#     print(f"成功处理数: {len(valid_results)}")
#     print(f"错误数: {error_count}")
    
#     if valid_results:
#         print(f"整体平均困惑度: {mean_ppl:.4f}")
#         print(f"最低困惑度: {min_ppl:.4f}")
#         print(f"最高困惑度: {max_ppl:.4f}")
    
#     # 保存到文件
#     if output_file:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(report, f, ensure_ascii=False, indent=2)
#         print(f"详细报告已保存到: {output_file}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='批量计算词汇困惑度')
    parser.add_argument('--input_file', help='输入数据文件路径 (JSON或JSONL格式)')
    parser.add_argument('-o', '--output', default='perplexity_results.json', 
                       help='输出文件路径 (默认: perplexity_results.json)')
    parser.add_argument('-f', '--format', choices=['json', 'jsonl'], default='json',
                       help='输出格式 (默认: json)')
    parser.add_argument('--checkpoint', help='检查点文件路径 (用于断点续传)')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='批处理大小 (默认: 10)')
    parser.add_argument('--sentence-batch-size', type=int, default=8,
                       help='句子批处理大小，用于GPU并行 (默认: 8)')
    parser.add_argument('--model', default='answerdotai/ModernBERT-large',
                       help='模型名称 (默认: answerdotai/ModernBERT-large)')
    parser.add_argument('--use-data-parallel', action='store_true',
                       help='启用单机多卡并行 (PyTorch DataParallel)')
    parser.add_argument('--fp16', action='store_true',
                       help='使用半精度推理（需要GPU支持）')
    parser.add_argument('--disable-batch-mode', action='store_true',
                       help='禁用批量处理模式（降低GPU利用率但更稳定）')
    #parser.add_argument('--report', help='汇总报告输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    # 加载数据
    logger.info(f"加载数据文件: {args.input_file}")
    try:
        if args.input_file.endswith('.jsonl'):
            data_items = load_jsonl_file(args.input_file)
        else:
            data_items = load_json_file(args.input_file)
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return
    
    logger.info(f"加载了 {len(data_items)} 个数据项")
    
    # 验证数据
    valid_items = [item for item in data_items if validate_data_item(item)]
    logger.info(f"有效数据项: {len(valid_items)}")
    
    if not valid_items:
        logger.error("没有有效的数据项")
        return
    
    # 初始化困惑度计算器
    logger.info(f"初始化模型: {args.model}")
    if args.use_data_parallel:
        logger.info("将启用 DataParallel 进行单机多卡推理")
    if args.fp16:
        logger.info("将使用 FP16 半精度推理（若GPU支持）")
    if not args.disable_batch_mode:
        logger.info(f"启用批量处理模式，句子批处理大小: {args.sentence_batch_size}")
    else:
        logger.info("批量处理模式已禁用（使用传统串行模式）")

    calculator = PerplexityCalculator(
        model_name=args.model,
        use_data_parallel=args.use_data_parallel,
        fp16=args.fp16,
    )
    
    # 批量处理
    logger.info("开始批量处理...")
    results = process_batch_with_checkpoint(
        calculator, 
        valid_items, 
        args.checkpoint, 
        args.batch_size,
        sentence_batch_size=args.sentence_batch_size,
        use_batch_mode=not args.disable_batch_mode
    )
    
    # 保存结果
    save_results(results, args.output, args.format)
    
    # 生成汇总报告
    #generate_summary_report(results, args.report)
    
    logger.info("批量处理完成！")

if __name__ == "__main__":
    main()