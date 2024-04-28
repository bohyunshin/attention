import torch
import torch.nn.functional as F

from attention.train.train_base import TrainBase



class Train(TrainBase):
    def __init__(self,
                 model_name,
                 epoch,
                 arg,
                 device,
                 smoothing=False):
        super().__init__(model_name=model_name,
                         epoch=epoch,
                         arg=arg,
                         device=device)
        self.smoothing = smoothing

    def cal_loss(self, pred, gold):
        # pred.shape = (batch_size, trg_vocab_size)
        # gold.shape = (batch_size, )

        if self.smoothing:
            eps = 0.1
            n_class = pred.size(1) # trg_vocab_size

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1,1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.arg.trg_pad_idx) # masking as True if not equal to trg_pad_idx
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=self.arg.trg_pad_idx)
        return loss

    def cal_performance(self, pred, gold):
        loss = self.cal_loss(pred, gold)

        pred = pred.max(1)[1] # max values, max indices > select max_indices
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.arg.trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word

    def get_data_from_batch(self, src, trg, src_pad_idx, trg_pad_idx, device):
        # src.shape = (src_seq_len, batch_size)
        # trg.shape = (trg_seq_len, batch_size)
        src_seq = self.patch_src(src, src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), self.patch_trg(trg, trg_pad_idx))
        return src_seq, trg_seq, gold

    def patch_src(self, src, pad_idx):
        # src.shape = (seq_len, batch_size)
        src = src.transpose(0, 1)
        return src

    def patch_trg(self, trg, pad_idx):
        trg = trg.transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        return trg, gold