from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer,max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 把 jsonl 文件读成一个"数据集对象"（Dataset）
        self.samples = load_dataset('json', data_files=data_path, split='train')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length-2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

if __name__ == "__main__":
    pass