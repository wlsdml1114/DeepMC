import torch
import pywt
import pandas as pd
import numpy as np

from scipy.interpolate import make_interp_spline

class DeepMCDataset(object):
    def __init__(self, file_path : str, predictors : list, target : str, seq_len : int, st_num : int = 90, levels :int = 5, RLat : float = 0.96, RLong : float = 1.5) -> None:
        try :
            temp = pd.read_csv(file_path)
        except :
            print('wrong file name')
        # predictors = z, target = y
        self.predictors = predictors
        self.target = target
        self.st_num = st_num
        self.levels = levels
        self.seq_len = seq_len
        self.RLat = RLat
        self.RLong = RLong

        try:
            self.Z = temp.loc[temp['지점번호']==self.st_num,self.predictors].values
            self.Y = temp.loc[temp['지점번호']==self.st_num,self.target].values
        except :
            print('st_num or predictor is wrong')
        # IOT sensor data/ Z is predictors + target / Y is target (to be predicted)

        # levels and length setting
        self.length = len(self.Z)

        # weather station prediction
        # Y_bar = weather_station_prediction
        self.Y_bar = self.Y - np.random.rand(self.length)

        # forecast error U
        self.U = self.Y - self.Y_bar
        
        # X = (Z,Y,{Rlat, Rlong})
        # WPD = (num_levels, lengths, X)
        self.WPD_x = np.zeros((self.levels,self.length,len(self.predictors)+3))
        self.WPD_u = np.zeros((self.levels,self.length))

        for j in range(len(self.predictors)):
            wptree = pywt.WaveletPacket(data=self.Z[:,j], wavelet='db1', mode='symmetric', maxlevel=self.levels)
            for i in range(1,6):
                levels = wptree.get_level(i, order = "freq") 
                data = self.avg(levels)

                times = np.mean(data)/np.mean(self.Z[:,j])

                data = data/times

                cubic_interploation_model=make_interp_spline(np.arange(0,self.length,2**(i)),data)
                
                xs=np.arange(self.length)
                ys=cubic_interploation_model(xs)

                self.WPD_x[i-1,:,j] = ys

        wptree = pywt.WaveletPacket(data=self.Y, wavelet='db1', mode='symmetric', maxlevel=self.levels)
        for i in range(1,6):
            levels = wptree.get_level(i, order = "freq") 
            data = self.avg(levels)

            times = np.mean(data)/np.mean(self.Y)

            data = data/times

            cubic_interploation_model=make_interp_spline(np.arange(0,self.length,2**(i)),data)
            
            xs=np.arange(self.length)
            ys=cubic_interploation_model(xs)

            self.WPD_x[i-1,:,-1] = ys
            
        self.WPD_x[:,:,-3] = self.RLong
        self.WPD_x[:,:,-2] = self.RLat

        wptree = pywt.WaveletPacket(data=self.U, wavelet='db1', mode='symmetric', maxlevel=self.levels)
        for i in range(1,6):
            levels = wptree.get_level(i, order = "freq") 
            data = self.avg(levels)

            times = np.mean(data)/np.mean(self.U)

            data = data/times

            cubic_interploation_model=make_interp_spline(np.arange(0,self.length,2**(i)),data)
            
            xs=np.arange(self.length)
            ys=cubic_interploation_model(xs)

            self.WPD_u[i-1,:] = ys

        print('aws data loading finish..')

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