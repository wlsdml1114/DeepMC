import torch
import pytorch_lightning as pl
from .dataset import DeepMCDataset
from typing import Tuple, Dict, Optional
class DeepMCDataLoader(pl.LightningDataModule):
    def __init__(self, file_path : str, predictor : list, target : str, seq_len : int, pred_len : int ,st_num : int = 90, levels : int = 5, batchsize : int = 16, num_workers : int =24):
        super().__init__()
        self.file_path = file_path
        self.predictor = predictor
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.st_num = st_num
        self.levels = levels
        self.batchsize = batchsize
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        maskrcnn_dataset = DeepMCDataset(
            file_path = self.file_path,
            predictors = self.predictor, 
            target = self.target, 
            seq_len = self.seq_len, 
            pred_len = self.pred_len ,
            st_num = self.st_num, 
            levels = self.levels, 
            RLat  = 0.96, # Rlat, RLong is calculated by station information
            RLong  = 1.5
        )
        train_size = int(0.95 * len(maskrcnn_dataset))
        test_size = len(maskrcnn_dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(maskrcnn_dataset, [train_size, test_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,batch_size = self.batchsize,num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,batch_size = self.batchsize,num_workers=self.num_workers) 