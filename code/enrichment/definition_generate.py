#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为合成词根据其构成字符的语义关系信息生成释义
输入：filtered_prediction_pairs.json
输出：compound_word_definitions.json

数据说明：
- head_id/tail_id：构成合成词的两个字符
- relation：字符间的语义关系（如：前缀、后缀、主谓等）
- 根据关系类型确定合成词的形式并生成相应释义
"""

import json
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径，以便导入Util模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Util.utils import load_client, get_response

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为合成词生成释义')
    parser.add_argument('--input-file', 
                       default='discriminate/rule_based_process/filtered_prediction_pairs.json',
                       help='输入的字符关系文件路径')
    parser.add_argument('--output-file', 
                       default='discriminate/definition_generate/word_with_definitions.json',
                       help='输出的合成词释义文件路径')
    parser.add_argument("--model", type=str, default='claude-sonnet-4@20250514', 
                       choices=['claude-opus-4@20250514', 'claude-sonnet-4@20250514', 
                               'claude-3-7-sonnet@20250219', 'claude-3-5-sonnet@20240620', 
                               'claude-3-5-haiku@20241022', 'gpt-4o', 'gpt-4o-mini', 
                               'gpt-4.1', 'gpt-4.1-nano', 'gpt-4.1-mini', 'o1', 'o4-mini', 
                               'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 
                               'gemini-2.0-flash'], 
                       help="选择使用的模型")
    parser.add_argument('--max-workers', type=int, default=2,
                       help='最大并发线程数')
    parser.add_argument('--max-retries', type=int, default=10,
                       help='API调用失败时的最大重试次数')
    parser.add_argument('--limit', type=int, default=None,
                       help='处理合成词数量限制，用于测试')
    
    return parser.parse_args()

class Args:
    """模型参数封装类"""
    def __init__(self, model, stream=True, think=False):
        self.model = model
        self.stream = stream
        self.think = think


def generate_definition_prompt(word_data):
    """为合成词生成释义的提示词"""
    system_prompt = """你是一位专业的词典编撰专家。现在你需要基于汉语二字词的一些信息，为汉语二字新词编撰准确、简洁的释义。
你获得的信息包括：前字的字形、前字的语素类（词性）、前字的释义、后字的字形、后字的语素类（词性）、后字的释义，以及它们之间的构词结构关系（如：定中、主谓等）。

任务要求：
1. 理解两个字通过给定的构词结构组成一个潜在的新词
2. 基于字的语素类（词性）、语义信息和它们之间的关系类型生成合成词的释义
3. 释义应该准确反映合成词的核心含义
4. 释义应该简洁明了，通常1-2句话，适用于词典
5. 直接输出释义内容，不需要额外说明

请根据以下信息生成合成词的释义："""

    head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
    tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
    
    # 根据关系类型确定合成词的形式
    relation = word_data.get('relation', '')


    user_prompt = f"""字符组合信息：
- 前字: {head_char}
- 前字字符词性: {word_data.get('head_pos', '')}
- 前字释义: {word_data.get('head_sense', '')}
- 后字: {tail_char}
- 后字词性: {word_data.get('tail_pos', '')}
- 后字释义: {word_data.get('tail_sense', '')}
- 构词结构: {relation}

该词({head_char+tail_char})的释义为：
"""

    return system_prompt, user_prompt

def query_definition(client, model_args, word_data, max_retries=3):
    """调用大模型生成词语释义"""
    system_prompt, user_prompt = generate_definition_prompt(word_data)
    
    message = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = get_response(
                client, 
                **vars(model_args), 
                system_prompt=system_prompt, 
                cur_message=message
            )
            
            # 清理响应内容
            definition = response.strip()
            
            # 获取字符
            head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
            tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
            
            # 根据关系类型确定合成词
            relation = word_data.get('relation', '')

            
            # 构建结果
            result = {
                'head_id': word_data.get('head_id'),
                'head_pos': word_data.get('head_pos'),
                'head_sense': word_data.get('head_sense'),
                'tail_id': word_data.get('tail_id'),
                'tail_pos': word_data.get('tail_pos'),
                'tail_sense': word_data.get('tail_sense'),
                'relation': relation,
                'confidence': word_data.get('confidence'),
                'definition': definition
            }

            print(f"成功生成释义: {head_char+tail_char} -> {definition[:50]}...")
            return result
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"API调用失败，正在重试 {retry_count}/{max_retries}: {e}")
                time.sleep(2 ** retry_count)  # 指数退避
            else:
                print(f"API调用最终失败: {e}")
                # 返回错误结果
                head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
                tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
                relation = word_data.get('relation', '')

                
                return {
                    'head_id': word_data.get('head_id'),
                    'head_pos': word_data.get('head_pos'),
                    'head_sense': word_data.get('head_sense'),
                    'tail_id': word_data.get('tail_id'),
                    'tail_pos': word_data.get('tail_pos'),
                    'tail_sense': word_data.get('tail_sense'),
                    'relation': relation,
                    'confidence': word_data.get('confidence'),
                    'definition': f"ERROR: {str(e)}"
                }

def process_words_parallel(word_pairs, model_args, client, max_workers=10):
    """并行处理合成词释义生成"""
    print(f"开始并行处理 {len(word_pairs)} 个合成词，使用 {max_workers} 个线程")
    
    all_results = []
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_word = {
                executor.submit(query_definition, client, model_args, word_data): word_data 
                for word_data in word_pairs
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_word), 
                             total=len(word_pairs), 
                             desc="生成合成词释义", 
                             unit="词"):
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    all_results.append(result)
                    
                except Exception as e:
                    word_data = future_to_word[future]
                    head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
                    tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
                    relation = word_data.get('relation', '')

                    
                    print(f"处理合成词失败 {head_char+tail_char}: {e}")
                    # 添加错误记录

    except KeyboardInterrupt:
        print("用户中断处理过程")
        return all_results
    except Exception as e:
        print(f"并行处理过程中发生错误: {e}")
        return all_results
    
    print(f"完成处理，共生成 {len(all_results)} 个结果")
    return all_results

def save_results(results, output_file):
    """保存结果到JSON文件"""
    print(f"保存结果到: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存 {len(results)} 个结果")

def get_unique_words(word_pairs):
    """获取唯一的词语列表（基于head_id、tail_id和relation组合去重）"""
    seen_combinations = set()
    unique_pairs = []
    
    for pair in word_pairs:
        head_id = pair.get('head_id', '')
        tail_id = pair.get('tail_id', '')
        relation = pair.get('relation', '')
        
        # 创建唯一标识符（head_id + tail_id + relation的组合）
        unique_key = f"{head_id}_{tail_id}_{relation}"
        
        if unique_key not in seen_combinations:
            seen_combinations.add(unique_key)
            unique_pairs.append(pair)
    
    print(f"去重后共有 {len(unique_pairs)} 个唯一的词语组合")
    return unique_pairs

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置模型参数
    model_args = Args(model=args.model, stream=True, think=False)
    
    # 加载客户端
    print(f"初始化模型客户端: {args.model}")
    client = load_client(model_args.model)
    
    # 处理文件路径
    input_file =  args.input_file
    output_file = args.output_file

    # 加载数据
    with open(input_file,'r',encoding='utf-8') as f:
        word_pairs = json.load(f)
    
    processed_word_pairs=[]
    if os.path.exists(output_file):
        with open(output_file,'r',encoding='utf-8') as f:
            processed_word_pairs = json.load(f)
        print(f"检测到已有 {len(processed_word_pairs)} 个已处理的词语，继续处理剩余词语")
        # 创建已处理词语的唯一标识集合
        processed_keys = set()
        for item in processed_word_pairs:
            head_id = item.get('head_id', '')
            tail_id = item.get('tail_id', '')
            relation = item.get('relation', '')
            unique_key = f"{head_id}_{tail_id}_{relation}"
            processed_keys.add(unique_key)
        
        # 过滤掉已处理的词语
        word_pairs = [
            wp for wp in word_pairs 
            if f"{wp.get('head_id', '')}_{wp.get('tail_id', '')}_{wp.get('relation', '')}" not in processed_keys
        ]
        print(f"剩余待处理词语数量: {len(word_pairs)}")


    

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_word = {
            executor.submit(query_definition, client, model_args, word_data, args.max_retries): word_data 
            for word_data in word_pairs
        }
        
        # 收集结果
        for future in tqdm(as_completed(future_to_word), 
                            total=len(word_pairs), 
                            desc="生成合成词释义", 
                            unit="词"):
            try:
                result = future.result(timeout=300)  # 5分钟超时
                processed_word_pairs.append(result)
                if len(processed_word_pairs) % 50 == 0:
                    save_results(processed_word_pairs, output_file)
                
            except Exception as e:
                word_data = future_to_word[future]
                head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
                tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
                relation = word_data.get('relation', '')

                
                print(f"处理合成词失败 {head_char+tail_char}: {e}")



    
    # 保存结果
    save_results(processed_word_pairs, output_file)
    
    # 统计结果
    success_count = sum(1 for r in processed_word_pairs if 'error' not in r and not r.get('generated_definition', '').startswith('ERROR:'))
    error_count = len(processed_word_pairs) - success_count

    print(f"处理完成！成功: {success_count}, 失败: {error_count}")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
