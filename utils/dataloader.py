import torch
import pytorch_lightning as pl
from .dataset import DeepMCDataset
from typing import Tuple, Dict, Optional

class DeepMCDataLoader(pl.LightningDataModule):
    def __init__(self, file_path : str, predictor : str, seq_len : int, st_num : int = 90, batchsize : int = 16, num_workers : int =24):
        super().__init__()
        self.file_path = file_path
        self.predictor = predictor
        self.seq_len = seq_len
        self.st_num = st_num
        self.batchsize = batchsize
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        maskrcnn_dataset = DeepMCDataset(self.file_path,self.predictor,self.seq_len,self.st_num)
        train_size = int(0.8 * len(maskrcnn_dataset))
        test_size = len(maskrcnn_dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(maskrcnn_dataset, [train_size, test_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize,num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize,num_workers=self.num_workers) 