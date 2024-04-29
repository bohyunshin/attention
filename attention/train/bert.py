import torch
from attention.train.train_base import TrainBase


class Train(TrainBase):
    def __init__(self,
                 model_name,
                 epoch,
                 arg,
                 device):
        super().__init__(model_name=model_name,
                         epoch=epoch,
                         arg=arg,
                         device=device)
        self.criterion = torch.nn.NLLLoss(ignore_index=0)

    def cal_loss(self, pred, gold):
        next_sent_output, mask_lm_output = pred

        next_loss = self.criterion(next_sent_output, gold["is_next"])
        mask_loss = self.criterion(mask_lm_output, gold["bert_label"])
        return next_loss + mask_loss

    def cal_performance(self, pred, gold):
        next_sent_output, mask_lm_output = pred
        loss = self.cal_loss(pred, gold)

        # next sentence prediction accuracy
        correct = next_sent_output.argmax(dim=-1).eq(gold["is_next"]).sum().item() # 0 or 1
        element = gold["is_next"].nelement() # 1

        return loss, correct, element


    def get_data_from_batch(self, batch, arg, device):
        data = {key: value.to(self.device) for key, value in batch.items()}
        model_input = {
            "x":data["bert_input"],
            "segment_label":data["segment_label"]
        }
        gold = {
            "bert_label":data["bert_label"],
            "is_next":data["is_next"]
        }
        return model_input, gold