import json
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_confidence_interval(count, n, confidence=0.95):
    """
    计算二项分布比例的置信区间 (Wilson Score Interval 或 Normal Approximation)
    这里使用简单的 Normal Approximation，适用于 n 较大且 p 不接近 0 或 1 的情况。
    对于论文严谨性，如果样本量小，建议用 Wilson Score。
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    
    p = count / n
    z = 1.96  # 95% confidence
    
    # Standard Error
    se = math.sqrt((p * (1 - p)) / n)
    margin_of_error = z * se
    
    lower = max(0.0, p - margin_of_error)
    upper = min(1.0, p + margin_of_error)
    
    return p, lower, upper, margin_of_error

def generate_latex_table(stats_data):
    """生成用于论文的 LaTeX 表格代码"""
    print("\n" + "="*20 + " LaTeX Table Code " + "="*20)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Category & Total Lines & Sampled & Valid Count & Validity Ratio (95\% CI) & Est. Valid Total \\")
    print(r"\midrule")
    
    for category, data in stats_data.items():
        p, lower, upper, margin = data['stats']
        # 格式化: 0.850 (+/- 0.03)
        ratio_str = f"{p:.3f} ($\\pm${margin:.3f})"
        print(f"{category.capitalize()} & {data['total_lines']:,} & {data['sampled_count']} & {data['valid_count']} & {ratio_str} & {data['estimated_total_valid']:,.0f} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Statistics of data validity checking using LLM. The validity ratio is estimated with 95\% confidence interval.}")
    print(r"\label{tab:data_validity}")
    print(r"\end{table}")
    print("="*60 + "\n")

def plot_results(stats_data, output_dir):
    """绘制可视化图表"""
    categories = list(stats_data.keys())
    ratios = [stats_data[c]['stats'][0] for c in categories]
    errors = [stats_data[c]['stats'][3] for c in categories]
    
    # 1. Validity Ratio Bar Chart with Error Bars
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, ratios, yerr=errors, capsize=10, color=['#4e79a7', '#f28e2b', '#e15759'], alpha=0.8)
    
    plt.title('Validity Ratio by Category (with 95% CI)', fontsize=14)
    plt.ylabel('Validity Ratio', fontsize=12)
    plt.ylim(0, 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2%}',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'validity_ratio_analysis.png'), dpi=300)
    print(f"Saved plot to {os.path.join(output_dir, 'validity_ratio_analysis.png')}")
    
    # 2. Estimated Total Count Comparison
    plt.figure(figsize=(8, 6))
    raw_counts = [stats_data[c]['total_lines'] for c in categories]
    est_counts = [stats_data[c]['estimated_total_valid'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, raw_counts, width, label='Raw Count', color='lightgray')
    rects2 = ax.bar(x + width/2, est_counts, width, label='Estimated Valid', color='#59a14f')
    
    ax.set_ylabel('Number of Sentences')
    ax.set_title('Raw vs. Estimated Valid Corpus Size')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height/1e6:.1f}M',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corpus_size_estimation.png'), dpi=300)
    print(f"Saved plot to {os.path.join(output_dir, 'corpus_size_estimation.png')}")

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM check results for paper")
    parser.add_argument("--result_file", type=str, default="discriminate\\final_data_1124\\corpus_check\\llm_check_results.json", help="Path to llm_check_results.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.result_file):
        print(f"Error: File {args.result_file} not found.")
        return

    with open(args.result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Process data
    stats_data = {}
    for category, data in results.items():
        p, lower, upper, margin = calculate_confidence_interval(data['valid_count'], data['sampled_count'])
        data['stats'] = (p, lower, upper, margin)
        stats_data[category] = data
        
        print(f"\nCategory: {category}")
        print(f"  Validity: {p:.2%} ± {margin:.2%}")
        print(f"  Raw Count: {data['total_lines']:,}")
        print(f"  Est. Valid: {data['estimated_total_valid']:,.0f}")

    # Generate outputs
    generate_latex_table(stats_data)
    
    output_dir = os.path.dirname(args.result_file)
    try:
        plot_results(stats_data, output_dir)
    except Exception as e:
        print(f"Could not generate plots (missing libraries?): {e}")

if __name__ == "__main__":
    main()
