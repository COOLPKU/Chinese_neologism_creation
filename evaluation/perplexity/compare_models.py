"""
对比掩码模型和自回归模型的计算方式

此脚本展示两种模型如何计算同一个词的困惑度
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ppl_model import PerplexityCalculator


def compare_models():
    """对比两种模型"""
    
    # 测试数据
    sample_data = {
        "head_id": "地02795_1_05_02",
        "tail_id": "雷05269_1_05_03",
        "sentences": [
            "民众应远离遗弃的～，以免发生爆炸。"
        ]
    }
    
    print("=" * 80)
    print("对比掩码模型和自回归模型")
    print("=" * 80)
    print(f"\n测试数据:")
    print(f"  目标词: 地雷")
    print(f"  例句: {sample_data['sentences'][0]}")
    print()
    
    # ========== 掩码模型 ==========
    print("=" * 80)
    print("1. 掩码模型 (Masked Model - BERT)")
    print("=" * 80)
    print("\n工作原理:")
    print("  1) 将目标词替换为 [MASK] token")
    print("     原句: 民众应远离遗弃的地雷，以免发生爆炸。")
    print("     掩码: 民众应远离遗弃的[MASK][MASK]，以免发生爆炸。")
    print()
    print("  2) 模型同时预测两个 [MASK] 位置的词")
    print("     预测第1个位置为'地'的概率: P(地|上下文)")
    print("     预测第2个位置为'雷'的概率: P(雷|上下文)")
    print()
    print("  3) 计算困惑度")
    print("     困惑度 = exp(-(log P(地) + log P(雷)) / 2)")
    print()
    
    try:
        print("初始化掩码模型...")
        masked_calculator = PerplexityCalculator(
            model_type="masked",
            fp16=True
        )
        
        print("计算中...")
        masked_result = masked_calculator.calculate_sentence_perplexities(sample_data)
        
        print(f"\n结果:")
        print(f"  困惑度: {masked_result['mean_perplexity']:.4f}")
        print()
    except Exception as e:
        print(f"错误: {e}\n")
    
    # ========== 自回归模型 ==========
    print("=" * 80)
    print("2. 自回归模型 (Autoregressive Model - GPT/Qwen)")
    print("=" * 80)
    print("\n工作原理:")
    print("  1) 第一步：预测第一个字'地'")
    print("     输入: 民众应远离遗弃的～，以免发生爆炸。～是：")
    print("     模型基于前文预测下一个token")
    print("     P(地|前文) = ?")
    print()
    print("  2) 第二步：预测第二个字'雷'")
    print("     输入: 民众应远离遗弃的～，以免发生爆炸。～是：地")
    print("     模型基于前文+已生成的'地'预测下一个token")
    print("     P(雷|前文,地) = ?")
    print()
    print("  3) 计算困惑度")
    print("     困惑度 = exp(-(log P(地) + log P(雷)) / 2)")
    print()
    
    # 注意：这里使用 gpt2 作为示例，实际使用时可以替换为其他模型
    print("注意: 使用 gpt2 作为示例（较小的模型，方便测试）")
    print("      实际应用中可以使用更大的中文模型，如 Qwen 系列")
    print()
    
    try:
        print("初始化自回归模型...")
        autoregressive_calculator = PerplexityCalculator(
            model_name="gpt2",  # 可替换为 "Qwen/Qwen2.5-7B" 等
            model_type="autoregressive",
            fp16=True
        )
        
        print("计算中（需要2次模型前向传播，一次预测一个字）...")
        autoregressive_result = autoregressive_calculator.calculate_sentence_perplexities(sample_data)
        
        print(f"\n结果:")
        print(f"  困惑度: {autoregressive_result['mean_perplexity']:.4f}")
        print()
    except Exception as e:
        print(f"错误: {e}\n")
    
    # ========== 对比总结 ==========
    print("=" * 80)
    print("总结：两种模型的主要区别")
    print("=" * 80)
    print("""
1. 计算方式:
   - 掩码模型: 双向上下文，同时预测所有被掩码的位置
   - 自回归模型: 单向生成，逐个字符顺序预测

2. 速度:
   - 掩码模型: 快（一次前向传播）
   - 自回归模型: 慢（需要 N 次前向传播，N 为目标词字符数）

3. 适用场景:
   - 掩码模型: 完形填空、上下文适配度评估
   - 自回归模型: 生成任务、序列预测

4. 困惑度含义:
   - 掩码模型: 词在双向上下文中的"意外程度"
   - 自回归模型: 词在生成过程中的"意外程度"

5. 注意事项:
   - 两种模型的困惑度值不能直接比较
   - 应在同一模型类型内进行相对比较
   - 选择模型应根据具体任务需求
    """)


if __name__ == "__main__":
    compare_models()
