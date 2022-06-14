import torch

from torch import nn

class LSTMstack(nn.Module):

    def __init__(self, num_encoder_feature : int):

        super().__init__()
        
        # Hyper parameter
        self.num_encoder_feature = num_encoder_feature

        # CNN stack before LSTM
        self.sequence= nn.Sequential(
            nn.Conv1d(self.num_encoder_feature, self.num_encoder_feature,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features= self.num_encoder_feature),
            nn.Conv1d(self.num_encoder_feature, self.num_encoder_feature,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=7)
        )

        # LSTM aftre CNN stack
        self.lstm = nn.LSTM(input_size = self.num_encoder_feature, hidden_size = self.num_encoder_feature, dropout = 0.2, num_layers = 2, batch_first = True, bidirectional = True)

    def forward(self, WPD):
        # cnnstack output
        cnn_output = self.sequence(WPD)

        # cnn output shape is (batch, feature, seq) / in this case [16, 7, 18]
        # but lstm input must be (batch, seq, feature) / in this case [16, 18, 7]
        # so we change axis of cnn output for match lstm input format
        new_cnn_output = torch.transpose(cnn_output, 1, 2)

        # lstm output is (output,(h_n, c_n))
        # output shape is (N,L,D∗H_out) when batch_first=True
        # h_n shape is (D∗num_layers,N,H_out), and it means the final hidden state for each element in the sequence
        # c_n shape is (D∗num_layers,N,H_cell), and it means the final cell state for each element in the sequence
        output, (h_n, c_n) = self.lstm(new_cnn_output)

        return output


class CNNstack(nn.Module):

    def __init__(self, num_encoder_feature):

        super().__init__()
        
        # Hyper parameter
        self.num_encoder_feature = num_encoder_feature

        # CNN stack
        self.sequence= nn.Sequential(
            nn.Conv1d(self.num_encoder_feature,self.num_encoder_feature,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.num_encoder_feature),
            nn.Conv1d(self.num_encoder_feature,self.num_encoder_feature,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.num_encoder_feature),
            nn.Conv1d(self.num_encoder_feature,self.num_encoder_feature,4),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, WPD):
        return self.sequence(WPD)