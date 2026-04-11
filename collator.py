import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.only_train_response = bool(cfg.dataset.only_train_response)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        group_ids = [d.get("group_id", 4) for d in batch]
        inputs["group_ids"] = torch.tensor(group_ids, dtype=torch.long)

        # Cross-attention routing: compute split position (early vs recent tokens)
        if any("early_history_text" in d for d in batch):
            early_texts = [d.get("early_history_text", "") for d in batch]
            early_enc = self.tokenizer(early_texts, add_special_tokens=False, padding=False)
            split_positions = [len(ids) for ids in early_enc["input_ids"]]
            inputs["history_split_pos"] = torch.tensor(split_positions, dtype=torch.long)

        return inputs



class TestCollator(object):

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        group_ids = [d.get("group_id", 4) for d in batch]
        inputs["group_ids"] = torch.tensor(group_ids, dtype=torch.long)

        # Cross-attention routing: compute split position (early vs recent tokens)
        if any("early_history_text" in d for d in batch):
            early_texts = [d.get("early_history_text", "") for d in batch]
            early_enc = self.tokenizer(early_texts, add_special_tokens=False, padding=False)
            split_positions = [len(ids) for ids in early_enc["input_ids"]]
            inputs["history_split_pos"] = torch.tensor(split_positions, dtype=torch.long)

        return (inputs, targets)