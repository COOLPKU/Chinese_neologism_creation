#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利用大模型API判断目标词及词义在上下文中是否合理，并预估整体比例。
"""

import json
import os
import sys
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径，以便导入Util模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
try:
    from Util.utils import load_client, get_response
except ImportError:
    # 如果直接运行脚本，可能需要调整路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
    from Util.utils import load_client, get_response

class Args:
    """模型参数封装类"""
    def __init__(self, model, stream=False, think=False):
        self.model = model
        self.stream = stream
        self.think = think

def parse_arguments():
    parser = argparse.ArgumentParser(description='利用大模型检查语料库上下文合理性')
    parser.add_argument('--corpus-dir', 
                       default='discriminate/final_data_1124/corpus_check/corpus_context',
                       help='包含jsonl文件的目录')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='每个文件抽取的样本数量')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro', 
                       help="选择使用的模型")
    parser.add_argument('--workers', type=int, default=50,
                       help='最大并发线程数')
    parser.add_argument('--output-file', 
                       default='discriminate/final_data_1124/corpus_check/llm_check_results.json',
                       help='结果保存路径')
    return parser.parse_args()

def generate_check_prompt(item):
    """生成检查提示词"""
    word = item.get('word', '')
    sense = item.get('sense', '')
    context = item.get('context', '')
    
    system_prompt = """你是一个语言学家，负责检查语料库中词语用法的正确性。
你需要判断给定的【目标词】在给定的【上下文】中是否使用恰当，且符合给定的【释义】。

请注意：
1. 上下文是从大规模语料中自动提取的，可能包含噪音或分词错误。
2. 如果目标词在上下文中是作为一个完整、合法的词出现的，且语义与给定释义一致（或相近），请判定为“合理”。
3. 如果目标词在上下文中只是其他词的一部分（例如“段时”出现在“一段时间”中），或者语义完全不符，请判定为“不合理”。
4. 仅回答 YES 或 NO。"""

    user_prompt = f"""
目标词: {word}
给定释义: {sense}
上下文: ...{context}...

请判断：该目标词在上述上下文中是否合理且符合给定释义？
回答 (YES/NO):
"""
    return system_prompt, user_prompt

def check_single_item(client, model_args, item):
    """检查单个样本"""
    system_prompt, user_prompt = generate_check_prompt(item)
    message = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    item_result = item.copy()
    
    try:
        # get_response(client, model, stream, think, system_prompt, cur_message)
        response = get_response(
            client, 
            model_args.model, 
            model_args.stream, 
            model_args.think, 
            system_prompt, 
            message
        )
        content = response.strip().upper()
        item_result['llm_response'] = content
        
        # 简单的解析逻辑
        if "YES" in content:
            item_result['is_valid'] = True
            return True, item_result
        elif "NO" in content:
            item_result['is_valid'] = False
            return False, item_result
        else:
            # 如果回答不明确，保守起见或者根据内容判断，这里简单处理
            item_result['is_valid'] = False
            return False, item_result
    except Exception as e:
        print(f"Error checking item: {e}")
        item_result['error'] = str(e)
        item_result['is_valid'] = False
        return False, item_result

def process_file(file_path, sample_size, client, model_args, workers):
    """处理单个文件"""
    print(f"Processing {file_path}...")
    
    # 1. 使用蓄水池抽样读取并抽样，避免内存溢出
    sampled_lines = []
    total_lines = 0
    random.seed(42)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Sampling {os.path.basename(file_path)}", unit="lines"):
            if not line.strip():
                continue
                
            if total_lines < sample_size:
                sampled_lines.append(line)
            else:
                # 蓄水池抽样: 以 k/n 的概率替换
                j = random.randint(0, total_lines)
                if j < sample_size:
                    sampled_lines[j] = line
            
            total_lines += 1
            
    print(f"  Total lines: {total_lines}")
    
    sampled_items = [json.loads(line) for line in sampled_lines]
    print(f"  Sampled {len(sampled_items)} items for checking.")
    
    # 3. 并发检查
    valid_count = 0
    checked_results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(check_single_item, client, model_args, item): item 
            for item in sampled_items
        }
        
        for future in tqdm(as_completed(future_to_item), total=len(sampled_items), desc="Checking"):
            try:
                is_valid, item_result = future.result()
                if is_valid:
                    valid_count += 1
                checked_results.append(item_result)
            except Exception as e:
                print(f"Task failed: {e}")
                
    # 4. 统计
    valid_ratio = valid_count / len(sampled_items) if sampled_items else 0
    estimated_total = total_lines * valid_ratio
    
    return {
        "file_name": os.path.basename(file_path),
        "total_lines": total_lines,
        "sampled_count": len(sampled_items),
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "estimated_total_valid": estimated_total,
        "checked_results": checked_results
    }

def main():
    args = parse_arguments()
    
    # 初始化客户端
    client = load_client(args.model)
    model_args = Args(model=args.model)
    
    results = {}
    
    # 假设的文件名映射
    file_mapping = {
        'existing': 'existing_words.jsonl',
        'positive': 'positive_words.jsonl',
        'negative': 'negative_words.jsonl'
    }
    
    corpus_dir = Path(args.corpus_dir)
    
    for category, filename in file_mapping.items():
        file_path = corpus_dir / filename
        
        # 如果找不到标准文件名，尝试查找包含关键词的文件（处理temp文件合并的情况）
        if not file_path.exists():
            # 尝试查找类似 existing_words_filtered.jsonl 或其他变体
            candidates = list(corpus_dir.glob(f"*{category}*.jsonl"))
            if candidates:
                # 优先选择最短的文件名（通常是合并后的）
                candidates.sort(key=lambda x: len(str(x)))
                file_path = candidates[0]
            else:
                print(f"Warning: Could not find file for category '{category}' in {corpus_dir}")
                continue
        
        stats = process_file(file_path, args.sample_size, client, model_args, args.workers)
        
        # 保存详细结果
        detailed_file = Path(args.output_file).parent / f"llm_check_details_{category}.jsonl"
        print(f"Saving detailed results to {detailed_file}...")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            for item in stats['checked_results']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 从统计中移除详细结果以减小最终json大小
        del stats['checked_results']
        
        results[category] = stats
        
    # 输出最终结果
    print("\n" + "="*50)
    print("Final Estimation Results")
    print("="*50)
    
    for category, stats in results.items():
        print(f"\nCategory: {category}")
        print(f"  File: {stats['file_name']}")
        print(f"  Total Lines (Raw): {stats['total_lines']:,}")
        print(f"  Sampled: {stats['sampled_count']:,}")
        print(f"  Valid (LLM Checked): {stats['valid_count']:,}")
        print(f"  Valid Ratio: {stats['valid_ratio']:.4f}")
        print(f"  Estimated Total Valid: {stats['estimated_total_valid']:,.0f}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
