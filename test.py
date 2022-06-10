import torch
import pywt

import pandas as pd
import numpy as np
import pytorch_lightning as pl

from typing import Tuple, Dict, Optional
from scipy.interpolate import make_interp_spline

from utils.dataloader import DeepMCDataLoader
from net.deepmc import DeepMC

dl = DeepMCDataLoader(
    file_path = '/home/ubuntu/jini1114/aws.csv',
    predictor = ['평균 기온', '최고 기온', '최저 기온'],
    target = '평균 기온', 
    seq_len = 24, 
    pred_len = 12
)
dl.setup()

for batch in dl.train_dataloader():
    test = batch
    break

model = DeepMC(
    num_encoder_hidden=7, 
    num_encoder_times=18,
    num_decoder_hidden = 20,
    num_decoder_times=12, 
    batch_size= 16, 
    num_of_CNN_stacks = 7,
    cnn_output_size = 105,
    num_feature = 3
)
output = model.forward(batch)
print(output)
print(output.shape)