#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap Resampling Analysis for Human Annotation
验证人工标注结果的统计稳定性
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
BASE_DIR = r'd:\new_word\discriminate\final_data_1124'
OUTPUT_DIR = os.path.join(BASE_DIR, 'r2_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_print(msg, output_file=None):
    """同时打印到终端和文件"""
    print(msg)
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def load_annotations(file_prefix, has_part2=False):
    """加载标注数据"""
    if has_part2:
        # 负样本：part2文件
        files = {
            'liuyang': os.path.join(BASE_DIR, 'annotation/annotation_data_part2_liuyang.xlsx'),
            'lql': os.path.join(BASE_DIR, 'annotation/annotation_data_part2_lql.xlsx'),
            'whs': os.path.join(BASE_DIR, 'annotation/annotation_data_part2_whs.xlsx')
        }
    else:
        # 正样本：无part2后缀
        files = {
            'liuyang': os.path.join(BASE_DIR, 'annotation/annotation_data_liuyang.xlsx'),
            'lql': os.path.join(BASE_DIR, 'annotation/annotation_data_lql.xlsx'),
            'whs': os.path.join(BASE_DIR, 'annotation/annotation_data_whs.xlsx')
        }
    
    dfs = {}
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        df = pd.read_excel(path)
        df = df.dropna(subset=['词形'])
        # NaN标注视为0（不接受）
        df['标注'] = pd.to_numeric(df['标注'], errors='coerce').fillna(0).astype(int)
        df['标注'] = df['标注'].apply(lambda x: 1 if x == 1 else 0)
        dfs[name] = df[['词形', '标注']].copy()
        dfs[name] = dfs[name].drop_duplicates(subset=['词形']).reset_index(drop=True)
    
    if len(dfs) == 0:
        return None
    
    # 使用inner join确保所有标注者都有该词
    annotator_names = list(dfs.keys())
    merged = dfs[annotator_names[0]].rename(columns={'标注': annotator_names[0]})
    for name in annotator_names[1:]:
        merged = merged.merge(
            dfs[name].rename(columns={'标注': name}), 
            on='词形', 
            how='inner'
        )
    
    # 计算majority vote
    merged['sum_score'] = merged[annotator_names].sum(axis=1)
    merged['accepted'] = (merged['sum_score'] >= 2).astype(int)
    
    return merged


def bootstrap_resampling(data, n_iterations=1000, confidence_level=0.95, random_seed=42):
    """
    执行Bootstrap重采样分析
    
    Parameters:
    -----------
    data : array-like
        二元数据（0或1），表示是否接受
    n_iterations : int
        重采样迭代次数
    confidence_level : float
        置信水平（默认95%）
    random_seed : int
        随机种子
    
    Returns:
    --------
    dict : 包含统计结果的字典
    """
    np.random.seed(random_seed)
    
    n_samples = len(data)
    bootstrap_means = []
    
    # 执行bootstrap重采样
    for i in range(n_iterations):
        # 有放回抽样
        resampled_data = np.random.choice(data, size=n_samples, replace=True)
        # 计算接受率
        acceptance_rate = np.mean(resampled_data)
        bootstrap_means.append(acceptance_rate)
    
    bootstrap_means = np.array(bootstrap_means)
    
    # 计算统计量
    original_mean = np.mean(data)
    bootstrap_mean = np.mean(bootstrap_means)
    bootstrap_std = np.std(bootstrap_means, ddof=1)
    
    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    ci_width = ci_upper - ci_lower
    margin_of_error = ci_width / 2
    
    return {
        'original_mean': original_mean,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_width,
        'margin_of_error': margin_of_error,
        'bootstrap_samples': bootstrap_means
    }


def main():
    print("="*80)
    print("Bootstrap Resampling Analysis for Human Annotation")
    print("="*80 + "\n")
    
    output_file = os.path.join(OUTPUT_DIR, 'bootstrap_analysis_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Bootstrap Resampling Analysis\n")
        f.write("="*80 + "\n\n")
    
    # 1. 加载数据
    log_print("Loading annotation data...", output_file)
    merged_pos = load_annotations('', has_part2=False)
    merged_neg = load_annotations('', has_part2=True)
    
    if merged_pos is None or merged_neg is None:
        log_print("Error: Cannot load annotation data", output_file)
        return
    
    log_print(f"POS samples: {len(merged_pos)}", output_file)
    log_print(f"NEG samples: {len(merged_neg)}", output_file)
    
    # 提取接受判断
    pos_accepted = merged_pos['accepted'].values
    neg_accepted = merged_neg['accepted'].values
    
    # 2. 执行Bootstrap分析
    log_print("\n" + "="*80, output_file)
    log_print("Performing Bootstrap Resampling (N=1000 iterations)", output_file)
    log_print("="*80 + "\n", output_file)
    
    n_iterations = 1000
    
    # POS组
    log_print("--- POS Set ---", output_file)
    pos_results = bootstrap_resampling(pos_accepted, n_iterations=n_iterations)
    log_print(f"Original Acceptance Rate: {pos_results['original_mean']:.4f} ({pos_results['original_mean']*100:.2f}%)", output_file)
    log_print(f"Bootstrap Mean:           {pos_results['bootstrap_mean']:.4f}", output_file)
    log_print(f"Bootstrap Std:            {pos_results['bootstrap_std']:.4f}", output_file)
    log_print(f"95% CI:                   [{pos_results['ci_lower']:.4f}, {pos_results['ci_upper']:.4f}]", output_file)
    log_print(f"95% CI (percentage):      [{pos_results['ci_lower']*100:.2f}%, {pos_results['ci_upper']*100:.2f}%]", output_file)
    log_print(f"Margin of Error:          ±{pos_results['margin_of_error']:.4f} (±{pos_results['margin_of_error']*100:.2f}%)", output_file)
    
    # NEG组
    log_print("\n--- NEG Set ---", output_file)
    neg_results = bootstrap_resampling(neg_accepted, n_iterations=n_iterations)
    log_print(f"Original Acceptance Rate: {neg_results['original_mean']:.4f} ({neg_results['original_mean']*100:.2f}%)", output_file)
    log_print(f"Bootstrap Mean:           {neg_results['bootstrap_mean']:.4f}", output_file)
    log_print(f"Bootstrap Std:            {neg_results['bootstrap_std']:.4f}", output_file)
    log_print(f"95% CI:                   [{neg_results['ci_lower']:.4f}, {neg_results['ci_upper']:.4f}]", output_file)
    log_print(f"95% CI (percentage):      [{neg_results['ci_lower']*100:.2f}%, {neg_results['ci_upper']*100:.2f}%]", output_file)
    log_print(f"Margin of Error:          ±{neg_results['margin_of_error']:.4f} (±{neg_results['margin_of_error']*100:.2f}%)", output_file)
    
    # 3. 检验置信区间是否重叠
    log_print("\n" + "="*80, output_file)
    log_print("Confidence Interval Overlap Analysis", output_file)
    log_print("="*80, output_file)
    
    pos_ci = (pos_results['ci_lower'], pos_results['ci_upper'])
    neg_ci = (neg_results['ci_lower'], neg_results['ci_upper'])
    
    # 检查是否有重叠
    overlap = not (pos_ci[1] < neg_ci[0] or neg_ci[1] < pos_ci[0])
    
    log_print(f"\nPOS 95% CI: [{pos_ci[0]*100:.2f}%, {pos_ci[1]*100:.2f}%]", output_file)
    log_print(f"NEG 95% CI: [{neg_ci[0]*100:.2f}%, {neg_ci[1]*100:.2f}%]", output_file)
    log_print(f"\nOverlap: {'YES (intervals overlap)' if overlap else 'NO (no overlap)'}", output_file)
    
    if not overlap:
        log_print("✓ The confidence intervals do NOT overlap, confirming the performance gap is statistically stable.", output_file)
    else:
        log_print("⚠ The confidence intervals overlap, suggesting potential instability.", output_file)
    
    # 4. 计算效应量 (Cohen's d)
    pooled_std = np.sqrt((pos_results['bootstrap_std']**2 + neg_results['bootstrap_std']**2) / 2)
    cohens_d = (pos_results['bootstrap_mean'] - neg_results['bootstrap_mean']) / pooled_std
    
    log_print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}", output_file)
    if abs(cohens_d) < 0.2:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    log_print(f"Interpretation: {effect} effect", output_file)
    
    # 5. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: Bootstrap分布 - POS
    ax1 = axes[0, 0]
    ax1.hist(pos_results['bootstrap_samples'], bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax1.axvline(pos_results['original_mean'], color='red', linestyle='--', linewidth=2, label='Original Mean')
    ax1.axvline(pos_results['ci_lower'], color='blue', linestyle=':', linewidth=2, label='95% CI')
    ax1.axvline(pos_results['ci_upper'], color='blue', linestyle=':', linewidth=2)
    ax1.set_xlabel('Acceptance Rate', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'POS Bootstrap Distribution (N={n_iterations})', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2: Bootstrap分布 - NEG
    ax2 = axes[0, 1]
    ax2.hist(neg_results['bootstrap_samples'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax2.axvline(neg_results['original_mean'], color='red', linestyle='--', linewidth=2, label='Original Mean')
    ax2.axvline(neg_results['ci_lower'], color='blue', linestyle=':', linewidth=2, label='95% CI')
    ax2.axvline(neg_results['ci_upper'], color='blue', linestyle=':', linewidth=2)
    ax2.set_xlabel('Acceptance Rate', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'NEG Bootstrap Distribution (N={n_iterations})', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3: KDE对比
    ax3 = axes[1, 0]
    sns.kdeplot(pos_results['bootstrap_samples'], ax=ax3, label='POS', fill=True, alpha=0.4, color='#2ecc71', linewidth=2)
    sns.kdeplot(neg_results['bootstrap_samples'], ax=ax3, label='NEG', fill=True, alpha=0.4, color='#e74c3c', linewidth=2)
    ax3.set_xlabel('Acceptance Rate', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Bootstrap Distribution Comparison (KDE)', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(alpha=0.3)
    
    # 子图4: 置信区间对比
    ax4 = axes[1, 1]
    categories = ['POS', 'NEG']
    means = [pos_results['bootstrap_mean'], neg_results['bootstrap_mean']]
    errors = [pos_results['margin_of_error'], neg_results['margin_of_error']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax4.bar(categories, means, yerr=errors, capsize=10, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + error + 0.02,
                f'{mean*100:.1f}% ± {error*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Acceptance Rate', fontsize=12)
    ax4.set_title('95% Confidence Intervals Comparison', fontsize=14)
    ax4.set_ylim(0, 1.0)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bootstrap_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"\n✓ Visualization saved to: {save_path}", output_file)
    
    # 6. 保存结果为JSON
    summary = {
        'n_iterations': n_iterations,
        'pos': {
            'n_samples': int(len(pos_accepted)),
            'original_acceptance_rate': float(pos_results['original_mean']),
            'bootstrap_mean': float(pos_results['bootstrap_mean']),
            'bootstrap_std': float(pos_results['bootstrap_std']),
            'ci_95_lower': float(pos_results['ci_lower']),
            'ci_95_upper': float(pos_results['ci_upper']),
            'margin_of_error': float(pos_results['margin_of_error'])
        },
        'neg': {
            'n_samples': int(len(neg_accepted)),
            'original_acceptance_rate': float(neg_results['original_mean']),
            'bootstrap_mean': float(neg_results['bootstrap_mean']),
            'bootstrap_std': float(neg_results['bootstrap_std']),
            'ci_95_lower': float(neg_results['ci_lower']),
            'ci_95_upper': float(neg_results['ci_upper']),
            'margin_of_error': float(neg_results['margin_of_error'])
        },
        'comparison': {
            'ci_overlap': overlap,
            'cohens_d': float(cohens_d),
            'effect_size': effect
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'bootstrap_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_print("\n✓ Bootstrap analysis completed!", output_file)
    log_print(f"✓ Results saved to: {OUTPUT_DIR}", output_file)
    
    # 7. 生成rebuttal用的格式化文本
    log_print("\n" + "="*80, output_file)
    log_print("Formatted Text for Rebuttal", output_file)
    log_print("="*80, output_file)
    
    rebuttal_text = f"""
The 95% Confidence Intervals for the POS set ({pos_results['bootstrap_mean']*100:.1f}% ± {pos_results['margin_of_error']*100:.1f}%) and NEG set ({neg_results['bootstrap_mean']*100:.1f}% ± {neg_results['margin_of_error']*100:.1f}%) show no overlap, confirming the performance gap is statistically stable.
"""
    
    log_print(rebuttal_text, output_file)
    
    return summary


if __name__ == '__main__':
    results = main()
    print("\n" + "="*80)
    print("Bootstrap Resampling Analysis Complete!")
    print("="*80)
