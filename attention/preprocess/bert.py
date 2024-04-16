import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie_conversations", required=True)
    parser.add_argument("--movie_lines", required=True)
    parser.add_argument("--raw_text", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return args

MAX_LEN = 64

class BERTDataset:
    def __init__(self,
                 movie_conversations,
                 movie_lines,
                 raw_text,
                 output):
        self.movie_conversations = movie_conversations
        self.movie_lines = movie_lines
        self.raw_text = raw_text
        self.output = output

        self.load_data()
        self.make_sentence_pair()
        self.make_text_file()

    def load_data(self):
        with open(self.movie_conversations, 'r', encoding='iso-8859-1') as c:
            self.conv = c.readlines()
        with open(self.movie_lines, 'r', encoding='iso-8859-1') as l:
            self.lines = l.readlines()

    def make_sentence_pair(self):
        ### splitting text using special lines
        lines_dic = {}
        for line in self.lines:
            objects = line.split(" +++$+++ ")
            lines_dic[objects[0]] = objects[-1]

        ### generate question answer pairs
        pairs = []
        for con in self.conv:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                qa_pairs = []

                if i == len(ids) - 1:
                    break

                first = lines_dic[ids[i]].strip()
                second = lines_dic[ids[i + 1]].strip()

                qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
                qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
                pairs.append(qa_pairs)
        self.pairs = pairs

    def make_text_file(self):
        text_data = []
        file_count = 0

        for sample in tqdm.tqdm([x[0] for x in self.pairs]):
            text_data.append(sample)

            # once we hit the 10K mark, save to file
            if len(text_data) == 10000:
                with open(os.path.join(self.raw_text, f"text_{file_count}.txt"), 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1
        self.text_data = text_data

    def train_tokenizer(self):
        paths = [str(x) for x in Path(self.raw_text).glob('**/*.txt')]

        ### training own tokenizer
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True
        )

        tokenizer.train(
            files=paths,
            vocab_size=30_000,
            min_frequency=5,
            limit_alphabet=1000,
            wordpieces_prefix='##',
            special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

        tokenizer.save_model(self.output, 'bert')
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.output, "bert-vocab.txt"), local_files_only=True)


if __name__ == "__main__":
    args = parse_args()

    bert_dataset = BERTDataset(args.movie_conversations,
                               args.movie_lines,
                               args.raw_text,
                               args.output)
    bert_dataset.train_tokenizer()
