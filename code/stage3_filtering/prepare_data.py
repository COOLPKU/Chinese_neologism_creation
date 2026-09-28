import json
import random
from tqdm import tqdm
import os

def load_word_senses(file_path):
    """
    加载 2_词.txt 文件，构建一个词到词义列表的映射。
    """
    word_senses = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # print(parts)
            if len(parts) >= 5:
                word_id_full, word, pinyin, pos, sense = parts[0], parts[1], parts[2], parts[3], parts[5]
                # 清理词形中的特殊字符
                if word not in word_senses:
                    word_senses[word] = []
                word_senses[word].append(sense)
    print(len(word_senses))
    return word_senses

def create_bem_data(processed_file, word_senses, output_dir):
    """
    从 3_MiCLS_processed.txt 创建 BEM 模型的训练、开发和测试数据。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    positive_examples = []
    with open(processed_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading positive examples"):
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                context, word_id_full, target_word, pos, sense = parts[0], parts[1], parts[2], parts[3], parts[4]
                # 将上下文中的占位符'～'替换为目标词
                positive_examples.append({
                    "context": context,
                    "target_word": target_word,
                    "positive_sense": sense
                })

    dataset = []
    for example in tqdm(positive_examples, desc="Generating negative examples"):
        target_word = example["target_word"]
        positive_sense = example["positive_sense"]
        
        # 查找负例
        if target_word in word_senses and len(word_senses[target_word]) > 1:
            possible_negative_senses = [s for s in word_senses[target_word] if s != positive_sense]
            if possible_negative_senses:
                for negative_sense in possible_negative_senses:
                # 负例
                    dataset.append({
                        "context": example["context"],
                        "target_word": target_word,
                        "sense": negative_sense,
                        "label": 0
                    })
                            # 正例
        dataset.append({
            "context": example["context"],
            "target_word": target_word,
            "sense": positive_sense,
            "label": 1
        })  

    # 打乱数据集
    random.shuffle(dataset)

    # 切分数据集
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))
    
    train_data = dataset[:train_size]
    dev_data = dataset[train_size:train_size + dev_size]
    test_data = dataset[train_size + dev_size:]

    # 保存文件
    with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
        for item in tqdm(train_data, desc="Writing train.jsonl"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(os.path.join(output_dir, 'dev.jsonl'), 'w', encoding='utf-8') as f:
        for item in tqdm(dev_data, desc="Writing dev.jsonl"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(os.path.join(output_dir, 'test.jsonl'), 'w', encoding='utf-8') as f:
        for item in tqdm(test_data, desc="Writing test.jsonl"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Data generation complete. Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    # 使用相对路径，假设此脚本在 discriminate/corpus_based_calculate/ 中运行
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..', '..') # 回到 new_word 目录

    word_senses_file = os.path.join(project_root, 'discriminate', 'data', '2_词.txt')
    processed_file = os.path.join(project_root, 'discriminate', 'data', '3_MiCLS_processed.txt')
    output_dir = os.path.join(project_root, 'discriminate', 'corpus_based_calculate', 'data')

    print("Loading word senses...")
    word_senses_map = load_word_senses(word_senses_file)
    
    print("Creating BEM data...")
    create_bem_data(processed_file, word_senses_map, output_dir)
