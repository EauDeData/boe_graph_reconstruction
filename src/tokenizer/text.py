import math
from typing import *
from transformers import AutoTokenizer
import torch

class BERTTokenizer:
    def __init__(self, context_length = 77):
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.sentence_len = context_length

        self.bos_token_id = 101
        self.eos_token_id = 102

    def __len__(self):
        return self.tokenizer.__len__()

    def tokenize(self, sentences: list):
        return torch.stack([torch.tensor(self.tokenizer.encode(sent, max_length=self.sentence_len, truncation=True,
                                                    padding='max_length',))
                           for sent in sentences])


    def decode(self, single_sentence: List[int]):
        return self.tokenizer.decode(single_sentence)