import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))

import sys
sys.path.append(os.getcwd())

import dill as pickle
import spacy
import torchtext.data
import torchtext.datasets
from torchtext.data import Dataset, BucketIterator
from constants import PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD
from attention.preprocess.preprocess_base import PreprocessBase


class Preprocess(PreprocessBase):
    def __init__(self,
                 lang_src,
                 lang_trg,
                 save_data,
                 data_src,
                 data_trg,
                 max_len,
                 min_word_count,
                 keep_case,
                 share_vocab):
        super().__init__()
        self.lang_src = lang_src
        self.lang_trg = lang_trg
        self.save_data = save_data
        self.data_src = data_src
        self.data_trg = data_trg
        self.max_len = max_len
        self.min_word_count = min_word_count
        self.keep_case = keep_case
        self.share_vocab = share_vocab

    def preprocess_raw_data(self):
        src_lang_model = spacy.load(self.lang_src)
        trg_lang_model = spacy.load(self.lang_trg)

        def tokenize_src(text):
            return [tok.text for tok in src_lang_model.tokenizer(text)]

        def tokenize_trg(text):
            return [tok.text for tok in trg_lang_model.tokenizer(text)]

        SRC = torchtext.data.Field(
            tokenize=tokenize_src, lower=not self.keep_case,
            pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

        TRG = torchtext.data.Field(
            tokenize=tokenize_trg, lower=not self.keep_case,
            pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

        MAX_LEN = self.max_len
        MIN_FREQ = self.min_word_count

        if not all([self.data_src, self.data_trg]):
            assert {self.lang_src, self.lang_trg} == {'de', 'en'}
        else:
            # Pack custom txt file into example datasets
            raise NotImplementedError

        def filter_examples_with_length(x):
            return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

        train, val, test = torchtext.datasets.Multi30k.splits(
            exts=('.' + self.lang_src, '.' + self.lang_trg),
            fields=(SRC, TRG),
            root="../.data",
            filter_pred=filter_examples_with_length
        )

        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        print('[Info] Get source language vocabulary size:', len(SRC.vocab))
        TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
        print('[Info] Get target language vocabulary size:', len(TRG.vocab))

        if self.share_vocab:
            print('[Info] Merging two vocabulary ...')
            for w, _ in SRC.vocab.stoi.items():
                # TODO: Also update the `freq`, although it is not likely to be used.
                if w not in TRG.vocab.stoi:
                    TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
            TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
            for w, i in TRG.vocab.stoi.items():
                TRG.vocab.itos[i] = w
            SRC.vocab.stoi = TRG.vocab.stoi
            SRC.vocab.itos = TRG.vocab.itos
            print('[Info] Get merged vocabulary size:', len(TRG.vocab))

        data = {
            'settings': self,
            'vocab': {'src': SRC, 'trg': TRG},
            'train': train.examples,
            'valid': val.examples,
            'test': test.examples}

        print('[Info] Dumping the processed data to pickle file', self.save_data)
        pickle.dump(data, open(self.save_data, 'wb'))

    def prepare_dataloader(self, arg, device):
        batch_size = arg.batch_size
        data = pickle.load(open(arg.data_pkl, "rb"))

        arg.max_token_seq_len = data["settings"].max_len
        arg.src_pad_idx = data["vocab"]["src"].vocab.stoi[PAD_WORD]
        arg.trg_pad_idx = data["vocab"]["trg"].vocab.stoi[PAD_WORD]

        arg.n_src_vocab = len(data["vocab"]["src"].vocab)
        arg.n_trg_vocab = len(data["vocab"]["trg"].vocab)

        # ========= Preparing Model =========#
        if arg.embs_share_weight:
            assert data["vocab"]["src"].vocab.stoi == data["vocab"]["trg"].vocab.stoi, \
                "To sharing word embedding the src/trg word2idx table shall be the same."

        fields = {"src": data["vocab"]["src"], "trg": data["vocab"]["trg"]}

        train = Dataset(examples=data["train"], fields=fields)
        val = Dataset(examples=data["valid"], fields=fields)

        train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
        val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

        return train_iterator, val_iterator, arg