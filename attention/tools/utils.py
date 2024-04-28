import pickle
import time
import torch
from torchtext.data.metrics import bleu_score


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1
    )).bool()
    return subsequent_mask


def save_pkl(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def get_bleu_score(output, gt, vocab, specials, max_n=4):

    def itos(x):
        x = list(x.cpu().numpy())
        tokens = vocab.lookup_tokens(x)
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(specials.keys()), tokens))
        return tokens

    pred = [out.max(dim=1)[1] for out in output]
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))

    score = bleu_score(pred_str, gt_str, max_n=max_n) * 100
    return  score


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(model.device)
    src_mask = model.make_src_mask(src).to(model.device)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len-1):
        memory = memory.to(model.device)
        tgt_mask = model.make_tgt_mask(ys).to(model.device)
        src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
        out = model.decode(ys, memory, tgt_mask, src_tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys


def asserting(expected, result):
    assert result == expected, f"Expected {expected}, got {result} instead"


def print_performances(header, ppl, accu, start_time, lr):
    print("  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, "\
          "elapse: {elapse:3.3f} min".format(
              header=f"({header})", ppl=ppl,
              accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))