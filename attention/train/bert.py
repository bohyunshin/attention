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
        loss = self.criterion(pred, gold)
        return loss

    def cal_performance(self, pred, gold):
        pass

    def get_data_from_batch(self, batch, arg, device):
        data = {key: value.to(self.device) for key, value in batch.items()}
        return data, data, data