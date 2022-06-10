import pywt
import torch 

import pandas as pd
import numpy as np

from scipy.interpolate import make_interp_spline

class DeepMCDataset(object):
    def __init__(self, file_path : str, predictors : list, target : str, seq_len : int, pred_len : int ,st_num : int = 90, levels :int = 5, RLat : float = 0.96, RLong : float = 1.5) -> None:
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
        self.pred_len = pred_len
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
        # WPD_x = (num_levels, X, lengths)
        # WPD_u = (num_levels, 1, lengths) / 1 for torch.stack
        self.WPD_x = np.zeros((self.levels,len(self.predictors)+3,self.length))
        self.WPD_u = np.zeros((self.levels,1,self.length))

        # higher level means more longer scale
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

                self.WPD_x[i-1,j,:] = ys

        wptree = pywt.WaveletPacket(data=self.Y, wavelet='db1', mode='symmetric', maxlevel=self.levels)
        for i in range(1,6):
            levels = wptree.get_level(i, order = "freq") 
            data = self.avg(levels)

            times = np.mean(data)/np.mean(self.Y)

            data = data/times

            cubic_interploation_model=make_interp_spline(np.arange(0,self.length,2**(i)),data)
            
            xs=np.arange(self.length)
            ys=cubic_interploation_model(xs)

            self.WPD_x[i-1,-3,:] = ys
            
        self.WPD_x[:,-1,:] = self.RLong
        self.WPD_x[:,-2,:] = self.RLat

        wptree = pywt.WaveletPacket(data=self.U, wavelet='db1', mode='symmetric', maxlevel=self.levels)
        for i in range(1,6):
            levels = wptree.get_level(i, order = "freq") 
            data = self.avg(levels)

            times = np.mean(data)/np.mean(self.U)

            data = data/times

            cubic_interploation_model=make_interp_spline(np.arange(0,self.length,2**(i)),data)
            
            xs=np.arange(self.length)
            ys=cubic_interploation_model(xs)

            self.WPD_u[i-1,0,:] = ys

        print('aws data loading finish..')

    def __getitem__(self, idx):
        # WPD_x, WPD_u is a input of deeplearning model
        # U is GT
        return (self.WPD_x[:,:,idx:idx+self.seq_len],
                self.WPD_u[:,:,idx:idx+self.seq_len],
                self.U[idx+self.seq_len:idx+self.seq_len+self.pred_len])
            

    def __len__(self):
        return self.length - self.seq_len - self.pred_len - 1

    def avg(self, levels):
        first = True
        for node in levels:
            if first:
                first = False
                sums = node.data
            else :
                sums = sums + node.data
        return sums 