#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为合成词生成例句
输入：包含定义的词语数据
输出：包含例句的词语数据

功能说明：
- 使用5个不同的大模型API为每个词造句
- 每个API为一个词造一个句子，总共3个句子
- 支持多进程并行处理
"""

import json
import os
import sys
import time
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径，以便导入Util模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Util.utils import load_client, get_response

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为合成词生成例句')
    parser.add_argument('--input-file', 
                       default='word_with_definitions.json',
                       help='输入的包含定义的词语文件路径')
    parser.add_argument('--output-file', 
                       default='words_with_sentences.json',
                       help='输出的包含例句的词语文件路径')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='最大并发线程数')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='API调用失败时的最大重试次数')
    parser.add_argument('--limit', type=int, default=None,
                       help='处理词语数量限制，用于测试')
    parser.add_argument('--sentences-per-word', type=int, default=3,
                       help='每个词生成的句子数量')
    
    return parser.parse_args()
MODELS = [
    'claude-3-7-sonnet@20250219',
    'gpt-4o',
    'gpt-4.1', 
    'gemini-2.5-pro',
    'claude-sonnet-4@20250514'
]


class Args:
    """模型参数封装类"""
    def __init__(self, model, stream=True, think=False):
        self.model = model
        self.stream = stream
        self.think = think
model_to_client = {
    'claude-3-7-sonnet@20250219': load_client('claude-3-7-sonnet@20250219'),
    'gpt-4o': load_client('gpt-4o'),
    'gpt-4.1': load_client('gpt-4.1'),
    'gemini-2.5-pro': load_client('gemini-2.5-pro'),
    'claude-sonnet-4@20250514': load_client('claude-sonnet-4@20250514')
}


def generate_sentence_prompt(word_data):
    """为合成词生成例句的提示词"""
    system_prompt = """你是一位专业的语言学家和例句编写专家。现在你需要为给定的汉语合成词创作自然、地道的例句。

任务要求：
1. 根据提供的词语和其释义，创作一个自然流畅的例句
2. 例句应该准确体现词语的含义和用法
3. 例句应该符合现代汉语的表达习惯
4. 例句长度适中，通常10-30个字
5. 直接输出例句，不需要额外说明或编号
6. 在例句中用～代替目标词。例如，词语是“电脑”，例句可以是“我用～工作和娱乐。”

请为以下词语创作一个例句："""

    compound_word = word_data['head_id'][0]+word_data['tail_id'][0]

    user_prompt = f"""词语：{compound_word}
释义：{word_data.get('definition', '')}


例句："""

    return system_prompt, user_prompt, compound_word

def query_sentence(client, model_args, word_data, max_retries=10):
    """调用大模型生成例句"""
    system_prompt, user_prompt, compound_word = generate_sentence_prompt(word_data)
    
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
            sentence = response.strip()
            
            print(f"成功生成例句 ({model_args.model}): {compound_word} -> {sentence[:30]}...")
            return {
                'model': model_args.model,
                'sentence': sentence
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"API调用失败，正在重试 {retry_count}/{max_retries} ({model_args.model}): {e}")
                time.sleep(2 ** retry_count)  # 指数退避
            else:
                print(f"API调用最终失败 ({model_args.model}): {e}")
                return {
                    'model': model_args.model,
                    'sentence': f"ERROR: {str(e)}"
                }

def generate_sentences_for_word(word_data, models_to_use, max_retries=3):
    """为单个词语使用多个模型生成例句"""
    all_sentences = []
    
    for model_name in models_to_use:
        try:
            # 为每个模型创建客户端和参数
            client = model_to_client[model_name]
            model_args = Args(model=model_name, stream=True, think=False)

            # 生成例句
            result = query_sentence(client, model_args, word_data, max_retries)
            all_sentences.append(result)
            
            # 避免API调用过于频繁
            time.sleep(0.5)
            
        except Exception as e:
            print(f"模型 {model_name} 处理失败: {e}")
            # 获取合成词
            head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
            tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
            all_sentences.append({
                'model': model_name,
                'sentence': f"ERROR: {str(e)}"
            })
    
    return [item['sentence'] for item in all_sentences]




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

def main():
    """主函数"""
    args = parse_arguments()
    
    # 处理文件路径
    input_file = os.path.join(os.path.dirname(__file__), args.input_file)
    output_file = os.path.join(os.path.dirname(__file__), args.output_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        all_word_data = json.load(f)

    processed_word_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_word_data = json.load(f)

    data_to_process = []
    processed_keys = set()
    for item in processed_word_data:
        head_id = item.get('head_id', '')
        tail_id = item.get('tail_id', '')
        relation = item.get('relation', '')
        key = f"{head_id}_{tail_id}_{relation}"
        processed_keys.add(key)
    
    for item in all_word_data:
        head_id = item.get('head_id', '')
        tail_id = item.get('tail_id', '')
        relation = item.get('relation', '')
        key = f"{head_id}_{tail_id}_{relation}"
        if key not in processed_keys:
            data_to_process.append(item)
    
    
    print(f"将使用以下模型: {MODELS[:args.sentences_per_word]}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 为每个词语提交任务
        future_to_word = {}

        for word_data in data_to_process:
            # 随机选择指定数量的模型
            
            future = executor.submit(
                generate_sentences_for_word, 
                word_data, 
                MODELS
            )
            future_to_word[future] = word_data
        
        # 收集结果
        for future in tqdm(as_completed(future_to_word), 
                            total=len(data_to_process), 
                            desc="生成例句", 
                            unit="词"):
            try:
                sentences = future.result(timeout=600)  # 10分钟超时
                word_data = future_to_word[future]
                
                # 合并词语数据和例句
                result = word_data.copy()
                result['sentences'] = sentences
                
                processed_word_data.append(result)
                if len(processed_word_data) % 10 == 0:
                    # 每处理10个词语保存一次结果
                    save_results(processed_word_data, output_file)
                
            except Exception as e:
                word_data = future_to_word[future]
                head_char = word_data.get('head_id', '')[0] if word_data.get('head_id') else ''
                tail_char = word_data.get('tail_id', '')[0] if word_data.get('tail_id') else ''
                
                print(f"处理词语失败 {head_char+tail_char}: {e}")
                
                # 添加错误记录
                result = word_data.copy()
                result['sentences'] = [{'model': 'ERROR', 'sentence': f"ERROR: {str(e)}", 'compound_word': f"{head_char+tail_char}"}]
                processed_word_data.append(result)
    # 保存结果
    save_results(processed_word_data, output_file)
    
    # 统计结果
    total_sentences = 0
    error_sentences = 0
    
    for result in processed_word_data:
        sentences = result.get('sentences', [])
        total_sentences += len(sentences)
        error_sentences += sum(1 for s in sentences if s.get('sentence', '').startswith('ERROR:'))
    
    success_sentences = total_sentences - error_sentences
    
    print(f"处理完成！")
    print(f"总词语数: {len(processed_word_data)}")
    print(f"总例句数: {total_sentences}")
    print(f"成功例句: {success_sentences}")
    print(f"失败例句: {error_sentences}")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
