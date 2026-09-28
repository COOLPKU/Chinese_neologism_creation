"""
分析GlossBERT筛选后的目标词数量和分布差异
比较 positive、negative、existing 三个文件
"""

import json
from collections import Counter, defaultdict
import os
from pathlib import Path


def load_and_analyze(file_path):
    """
    加载jsonl文件并分析目标词
    
    Returns:
        dict: 包含统计信息的字典
    """
    print(f"\n正在分析: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    word_counter = Counter()
    word_contexts = defaultdict(list)
    word_scores = defaultdict(list)
    total_samples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            try:
                item = json.loads(line.strip())
                word = item.get('word', '')
                context = item.get('context', '')
                score = item.get('probability', item.get('score', 0))
                
                word_counter[word] += 1
                word_contexts[word].append(context)
                word_scores[word].append(score)
                
            except json.JSONDecodeError:
                continue
    
    print(f"  总样本数: {total_samples:,}")
    print(f"  唯一目标词数: {len(word_counter):,}")
    
    # 计算平均置信度
    avg_scores = {word: sum(scores)/len(scores) 
                  for word, scores in word_scores.items()}
    
    return {
        'file_name': os.path.basename(file_path),
        'total_samples': total_samples,
        'unique_words': len(word_counter),
        'word_counter': word_counter,
        'word_contexts': word_contexts,
        'word_scores': word_scores,
        'avg_scores': avg_scores
    }


def compare_distributions(positive_data, negative_data, existing_data):
    """
    比较三个文件的分布差异
    """
    print("\n" + "="*80)
    print("分布比较分析")
    print("="*80)
    
    # 基本统计
    print("\n【基本统计】")
    print(f"{'类别':<15} {'总样本数':>12} {'唯一词数':>12} {'平均每词样本数':>15}")
    print("-" * 80)
    
    for name, data in [('Positive', positive_data), 
                       ('Negative', negative_data), 
                       ('Existing', existing_data)]:
        if data:
            avg_samples = data['total_samples'] / data['unique_words'] if data['unique_words'] > 0 else 0
            print(f"{name:<15} {data['total_samples']:>12,} {data['unique_words']:>12,} {avg_samples:>15.2f}")
    
    # 词汇重叠分析
    print("\n【词汇重叠分析】")
    pos_words = set(positive_data['word_counter'].keys()) if positive_data else set()
    neg_words = set(negative_data['word_counter'].keys()) if negative_data else set()
    exist_words = set(existing_data['word_counter'].keys()) if existing_data else set()
    
    print(f"Positive ∩ Negative: {len(pos_words & neg_words):,} 个词")
    print(f"Positive ∩ Existing: {len(pos_words & exist_words):,} 个词")
    print(f"Negative ∩ Existing: {len(neg_words & exist_words):,} 个词")
    print(f"三者共有: {len(pos_words & neg_words & exist_words):,} 个词")
    
    print(f"\nPositive 独有: {len(pos_words - neg_words - exist_words):,} 个词")
    print(f"Negative 独有: {len(neg_words - pos_words - exist_words):,} 个词")
    print(f"Existing 独有: {len(exist_words - pos_words - neg_words):,} 个词")
    
    # Top 20 高频词
    print("\n【Top 20 高频词】")
    for name, data in [('Positive', positive_data), 
                       ('Negative', negative_data), 
                       ('Existing', existing_data)]:
        if data:
            print(f"\n{name}:")
            top_words = data['word_counter'].most_common(20)
            for i, (word, count) in enumerate(top_words, 1):
                avg_score = data['avg_scores'].get(word, 0)
                print(f"  {i:2d}. {word:<10} 次数: {count:>6,}  平均分: {avg_score:.4f}")
    
    # 置信度分析
    print("\n【置信度统计】")
    print(f"{'类别':<15} {'平均置信度':>12} {'最小置信度':>12} {'最大置信度':>12}")
    print("-" * 80)
    
    for name, data in [('Positive', positive_data), 
                       ('Negative', negative_data), 
                       ('Existing', existing_data)]:
        if data and data['avg_scores']:
            all_scores = [score for scores in data['word_scores'].values() for score in scores]
            avg_score = sum(all_scores) / len(all_scores)
            min_score = min(all_scores)
            max_score = max(all_scores)
            print(f"{name:<15} {avg_score:>12.4f} {min_score:>12.4f} {max_score:>12.4f}")
    
    # 样本数分布
    print("\n【样本数分布】")
    for name, data in [('Positive', positive_data), 
                       ('Negative', negative_data), 
                       ('Existing', existing_data)]:
        if data:
            counts = list(data['word_counter'].values())
            print(f"\n{name}:")
            print(f"  1个样本的词: {sum(1 for c in counts if c == 1):,} 个")
            print(f"  2-5个样本的词: {sum(1 for c in counts if 2 <= c <= 5):,} 个")
            print(f"  6-10个样本的词: {sum(1 for c in counts if 6 <= c <= 10):,} 个")
            print(f"  11-50个样本的词: {sum(1 for c in counts if 11 <= c <= 50):,} 个")
            print(f"  50+个样本的词: {sum(1 for c in counts if c > 50):,} 个")


def save_summary(positive_data, negative_data, existing_data, output_file):
    """保存分析摘要到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("GlossBERT筛选结果分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 基本统计
        f.write("基本统计:\n")
        f.write("-"*80 + "\n")
        for name, data in [('Positive', positive_data), 
                           ('Negative', negative_data), 
                           ('Existing', existing_data)]:
            if data:
                f.write(f"{name}:\n")
                f.write(f"  总样本数: {data['total_samples']:,}\n")
                f.write(f"  唯一词数: {data['unique_words']:,}\n")
                f.write(f"  平均每词: {data['total_samples']/data['unique_words']:.2f}\n\n")
        
        # 词汇重叠
        f.write("\n词汇重叠:\n")
        f.write("-"*80 + "\n")
        pos_words = set(positive_data['word_counter'].keys()) if positive_data else set()
        neg_words = set(negative_data['word_counter'].keys()) if negative_data else set()
        exist_words = set(existing_data['word_counter'].keys()) if existing_data else set()
        
        f.write(f"Positive ∩ Negative: {len(pos_words & neg_words):,}\n")
        f.write(f"Positive ∩ Existing: {len(pos_words & exist_words):,}\n")
        f.write(f"Negative ∩ Existing: {len(neg_words & exist_words):,}\n")
        f.write(f"三者共有: {len(pos_words & neg_words & exist_words):,}\n\n")
        
        f.write(f"Positive 独有: {len(pos_words - neg_words - exist_words):,}\n")
        f.write(f"Negative 独有: {len(neg_words - pos_words - exist_words):,}\n")
        f.write(f"Existing 独有: {len(exist_words - pos_words - neg_words):,}\n")
    
    print(f"\n分析报告已保存到: {output_file}")


def main():
    # 文件路径
    base_dir = Path(__file__).parent
    positive_file = base_dir / 'positive_words_filtered.jsonl'
    negative_file = base_dir / 'negative_words_filtered.jsonl'
    existing_file = base_dir / 'existing_words_filtered.jsonl'
    
    # 加载并分析数据
    positive_data = load_and_analyze(positive_file)
    negative_data = load_and_analyze(negative_file)
    existing_data = load_and_analyze(existing_file)
    
    # 比较分布
    compare_distributions(positive_data, negative_data, existing_data)
    
    # 保存摘要
    output_file = base_dir / 'analysis_summary.txt'
    save_summary(positive_data, negative_data, existing_data, output_file)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
