import torch
import pywt
import pandas as pd
import numpy as np

from scipy.interpolate import make_interp_spline

class DeepMCDataset(object):
    def __init__(self, file_path : str, predictor : str, seq_len : int, st_num : int = 90) -> None:
        try :
            temp = pd.read_csv(file_path)
        except :
            print('wrong file name')
        try:
            self.aws = temp.loc[temp['지점번호']==st_num,predictor].values
        except :
            print('st_num or predictor is wrong')

        wptree = pywt.WaveletPacket(data=self.aws, wavelet='db1', mode='symmetric', maxlevel=5)

        temp = []

        for i in range(1,6):
            levels = wptree.get_level(i, order = "freq") 

            data = self.avg(levels)

            times = np.mean(data)/np.mean(self.aws)

            data = data/times

            cubic_interploation_model=make_interp_spline(np.arange(0,len(self.aws),2**(i)),data)

            xs=np.arange(len(self.aws))
            ys=cubic_interploation_model(xs)

            temp.append(torch.tensor(ys).float())

        print('aws data loading finish..')
        self.WPD = temp
        self.seq_len = seq_len

    def __getitem__(self, idx):
        return (self.WPD[0][idx:idx+self.seq_len],
        self.WPD[1][idx:idx+self.seq_len],
        self.WPD[2][idx:idx+self.seq_len],
        self.WPD[3][idx:idx+self.seq_len],
        self.WPD[4][idx:idx+self.seq_len],
        self.aws[idx+self.seq_len:idx+self.seq_len+self.seq_len]
        )

    def __len__(self):
        return len(self.aws) - self.seq_len - 24

    def avg(self, levels):
        first = True
        for node in levels:
            if first:
                first = False
                sums = node.data
            else :
                sums = sums + node.data
        return sums 