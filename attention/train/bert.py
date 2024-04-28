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

    def cal_loss(self, pred, gold):
        pass

    def cal_performance(self, pred, gold):
        pass

    def get_data_from_batch(self, src, trg, src_pad_idx, trg_pad_idx, device):
        pass