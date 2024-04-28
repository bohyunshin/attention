import torch
from abc import abstractmethod
from tqdm import tqdm
import importlib
import time
import math
import os

from attention.tools.utils import print_performances


class TrainBase:
    def __init__(self,
                 model_name,
                 optimizer,
                 epoch,
                 arg,
                 device):
        # init model
        model = importlib.import_module(f"attention.models.{model_name}").Model
        self.model = model(**arg).to(device)
        self.optimizer = optimizer
        self.epoch = epoch
        self.arg = arg
        self.device = device

    @abstractmethod
    def cal_loss(self, pred, gold):
        raise ValueError("cal_loss method should be implemented")

    @abstractmethod
    def cal_performance(self, pred, gold):
        raise ValueError("cal_performance method should be implemented")

    @abstractmethod
    def get_data_from_batch(self, src, trg, src_pad_idx, trg_pad_idx, device):
        raise ValueError("get_data_from_batch method should be implemented")

    @abstractmethod
    def prepare_dataloaders(self, arg, device):
        raise ValueError("prepare_dataloaders should be implemented")

    def train_epoch(self, train_data):
        # set train mode
        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        for batch in tqdm(train_data, mininterval=2):
            src_seq, trg_seq, gold = self.get_data_from_batch(batch.src, batch.trg, self.arg.src_pad_idx, self.arg.trg_pad_idx, self.device)
            model_input = {
                "src_seq": src_seq,
                "trg_seq": trg_seq
            }

            # forward
            self.optimizer.zero_grad()
            pred = self.model(**model_input)

            # backward and update parameters
            loss, n_correct, n_word = self.cal_performance(pred, gold)
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy


    def eval_epoch(self, validation_data):
        # validation mode
        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2):
                src_seq, trg_seq, gold = self.get_data_from_batch(batch.src, batch.trg, self.arg.src_pad_idx,
                                                                  self.arg.trg_pad_idx, self.device)

                # forward
                pred = self.model(src_seq, trg_seq)
                loss, n_correct, n_word = self.cal_performance(pred, gold)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def train(self, train_data, validation_data):
        valid_losses = []
        for epoch_i in range(self.epoch):
            print(f"[ Epoch {epoch_i}]")

            start = time.time()
            train_loss, train_accuracy = self.train_epoch(train_data)
            train_ppl = math.exp(min(train_loss, 100))

            # current learning rate
            lr = self.optimizer._optimizer.param_groups[0]["lr"]
            print_performances("Training", train_ppl, train_accuracy, start, lr)

            start = time.time()
            valid_loss, valid_accuracy = self.eval_epoch(validation_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performances("Validation", valid_ppl, valid_accuracy, start, lr)

            valid_losses += [valid_loss]

            checkpoint = {"epoch": epoch_i, "settings": self.arg, "model": self.model.state_dict()}

            if self.arg.save_mode == "all":
                model_name = "model_accu_{accu:3.3f}.chkpt".format(accu=100 * valid_accuracy)
                torch.save(checkpoint, model_name)
            elif self.arg.save_mode == "best":
                model_name = "model.chkpt"
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, os.path.join(self.arg.output_dir, model_name))
                    print("    - [Info] The checkpoint file has been updated.")