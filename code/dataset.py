import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ExplanationDataset(Dataset):
    def __init__(self, data_list, tokenizer, processor, max_length=512):
        """
        data_list: List of dicts [{'image_path': '...', 'text': '...'}]
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load and Process Image
        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # 2. Tokenize Text (The Explanation)
        # We append an EOS token so the model learns to stop
        text = item['text'] + self.tokenizer.eos_token
        
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        
        # Labels are same as input_ids for causal LM training
        # (The masking of padding is handled by setting labels to -100)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }