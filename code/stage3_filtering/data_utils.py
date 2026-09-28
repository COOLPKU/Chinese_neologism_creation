
import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer
import os

class WSDDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        sense = item['sense']
        label = item['label']

        encoding = self.tokenizer.encode_plus(
            text=context,
            text_pair=sense,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def create_data_loader(file_path, tokenizer, batch_size, max_length=512, shuffle=True):
    dataset = WSDDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

if __name__ == '__main__':
    # 这是一个测试
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 假设你的项目结构是 new_word/discriminate/corpus_based_calculate/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'train.jsonl')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run prepare_data.py first.")
    else:
        train_loader = create_data_loader(data_path, tokenizer, batch_size=2)
        
        for batch in train_loader:
            print("Input IDs:", batch['input_ids'].shape)
            print("Attention Mask:", batch['attention_mask'].shape)
            print("Token Type IDs:", batch['token_type_ids'].shape)
            print("Labels:", batch['labels'].shape)
            # 打印一个样本的解码结果
            print("Sample Decoded:", tokenizer.decode(batch['input_ids'][0]))
            break
