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
from attention.preprocess.preprocess_base import PreprocessBase


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie_conversations", required=False)
    parser.add_argument("--movie_lines", required=False)
    parser.add_argument("--raw_text", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--batch_size", required=False, type=int)
    args = parser.parse_args()
    return args


class BERTDataset:
    def __init__(self,
                 seq_len,
                 movie_conversations,
                 movie_lines,
                 raw_text,
                 output,
                 mode):
        self.seq_len = seq_len
        # self.movie_conversations = movie_conversations
        self.movie_lines = movie_lines
        self.raw_text = raw_text
        self.output = output
        self.mode = mode
        lines = self.load_data()
        self.make_sentence_pair(movie_conversations, lines)
        self.make_text_file()

    def load_data(self):
        # with open(self.movie_conversations, 'r', encoding='iso-8859-1') as c:
        #     conv = c.readlines()
        with open(self.movie_lines, 'r', encoding='iso-8859-1') as l:
            lines = l.readlines()
        return lines

    def make_sentence_pair(self, conv, lines):
        ### splitting text using special lines
        lines_dic = {}
        for line in lines:
            objects = line.split(" +++$+++ ")
            lines_dic[objects[0]] = objects[-1]

        ### generate question answer pairs
        pairs = []
        for con in conv:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                qa_pairs = []

                if i == len(ids) - 1:
                    break

                first = lines_dic[ids[i]].strip()
                second = lines_dic[ids[i + 1]].strip()

                qa_pairs.append(' '.join(first.split()[:self.seq_len]))
                qa_pairs.append(' '.join(second.split()[:self.seq_len]))
                pairs.append(qa_pairs)
        self.lines = pairs

    def make_text_file(self):
        text_data = []
        file_count = 0

        for sample in tqdm.tqdm([x[0] for x in self.lines]):
            text_data.append(sample)

            # once we hit the 10K mark, save to file
            if len(text_data) == 10000:
                with open(os.path.join(self.raw_text, self.mode, f"text_{file_count}.txt"), 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1

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

        tokenizer.save_model(os.path.join(self.output, self.mode), 'bert')
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.output, self.mode, "bert-vocab.txt"), local_files_only=True)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        # step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # step 3
        t1 = [self.tokenizer.cls_token_id] + t1_random + [self.tokenizer.sep_token_id]
        t2 = t2_random + [self.tokenizer.sep_token_id]
        t1_label = [self.tokenizer.pad_token_id] + t1_label + [self.tokenizer.pad_token_id]
        t2_label = t2_label + [self.tokenizer.pad_token_id]

        # step 4
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.pad_token_id for _ in range(self.seq_len - len(bert_input))]
        bert_input += padding
        bert_label += padding
        segment_label += padding

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)["input_ids"][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab["[MASK]"])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label


    def get_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]


class Preprocess(PreprocessBase):
    def __init__(self,
                 max_len,
                 movie_conversations,
                 movie_lines,
                 raw_text,
                 output,
                 **kwargs):
        super().__init__()
        self.seq_len = max_len
        self.movie_conversations = movie_conversations
        self.movie_lines = movie_lines
        self.raw_text = raw_text
        self.output = output

    def preprocess_raw_data(self):
        with open(self.movie_conversations, 'r', encoding='iso-8859-1') as l:
            conv = l.readlines()
        n_test = int(len(conv) * 0.7)
        train_conv = conv[:n_test]
        val_conv = conv[n_test:]

        train_dataset = BERTDataset(
            self.seq_len,
            train_conv,
            self.movie_lines,
            self.raw_text,
            self.output,
            "train"
        )
        train_dataset.train_tokenizer()

        validation_dataset = BERTDataset(
            self.seq_len,
            val_conv,
            self.movie_lines,
            self.raw_text,
            self.output,
            "val"
        )
        validation_dataset.train_tokenizer()

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def prepare_dataloader(self, arg, device):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=arg.batch_size,
            shuffle=True,
            pin_memory=True
        )
        validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=arg.batch_size,
            shuffle=True,
            pin_memory=True
        )
        arg.n_vocab = len(self.train_dataset.tokenizer.vocab)

        return train_dataloader, validation_dataloader, arg
