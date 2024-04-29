from abc import abstractmethod


class PreprocessBase:
    def __init__(self):
        pass

    @abstractmethod
    def preprocess_raw_data(self):
        raise ValueError("preprocess_raw_data method should be implemented")

    @abstractmethod
    def prepare_dataloader(self, arg, device):
        raise ValueError("prepare_dataloader method should be implemented")
