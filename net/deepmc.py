import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import dropout, nn

class Attention_Decoder(nn.Module):
    def __init__(self,num_encoder_hidden : int , num_encoder_times : int, num_decoder_hidden : int,num_decoder_times : int):
        super().__init__()
        self.num_encoder_hidden = num_encoder_hidden
        self.num_encoder_times = num_encoder_times
        self.num_decoder_hidden = num_decoder_hidden
        self.num_decoder_times = num_decoder_times

        self.W_a = nn.Linear(self.num_decoder_hidden, self.num_decoder_times)
        self.U_a = nn.Linear(self.num_encoder_hidden, self.num_encoder_times)
        self.v_a = nn.Linear(self.num_encoder_times+self.num_decoder_times,1)
        self.phi_weight = nn.Linear(self.num_encoder_times, self.num_encoder_hidden)

        self.Decoder = nn.LSTMCell(self.num_encoder_hidden, self.num_decoder_hidden)
    
    def forward(self, LSTM, CNNs):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12

        # Position based content attention layer
        # s_i / (H', 1)
        s_i = torch.rand((self.num_decoder_hidden))
        v_a_output = []
        for j in range(self.num_encoder_times):
                
            # delta_i_j / (T, 1)
            delta_i_j = torch.zeros((self.num_encoder_times))
            delta_i_j[:j] = 1

            # phi_delta / (H, 1)
            phi_delta = self.phi_weight(delta_i_j)
            # phi_delta_hadamard / (H, 1)
            phi_delta_hadamard = phi_delta*LSTM[j]

            # U_a_output / (T, 1)
            U_a_output = self.U_a(phi_delta_hadamard)
            
            # W_a_output / (T', 1)
            W_a_output = self.W_a(s_i)

            # concat / (T'+T, 1)
            concat = torch.cat((W_a_output,U_a_output))

            # delta_i_t_j / (T+T', 1)
            delta_i_T_j = torch.zeros((self.num_encoder_times+self.num_decoder_times))
            delta_i_T_j[i+self.num_encoder_times-j] = 1

            # concat_tanh / (T'+T, 1)
            concat_tanh = torch.tanh(concat)

            # concat_tanh_hadamard_delta / (T'+T, 1)
            concat_tanh_hadamard_delta = concat_tanh*delta_i_T_j

            # v_a_output / scalar
            v_a_output.append(self.v_a(concat_tanh_hadamard_delta))
        
        # e_ij / (T, 1)
        e_ij = torch.cat(v_a_output)

        # a_ij / (T, 1)
        a_ij = torch.softmax(e_ij,dim=0)

        a_ij = a_ij.unsqueeze(1)
        # a_ij / (1, T)
        a_ij = a_ij.permute(1,0)

        # c_i = (1, H)
        c_i = torch.mm(a_ij,LSTM)

        # Scaled Guided Attention Layer

        return c_i

class LSTMstack(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence= nn.Sequential(
            nn.Conv1d(7,7,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=7),
            nn.Conv1d(7,7,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=7)
        )
        self.lstm = nn.LSTM(input_size = 7, hidden_size = 7, dropout = 0.2, num_layers = 2, batch_first = True)

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
    def __init__(self):
        super().__init__()
        self.sequence= nn.Sequential(
            nn.Conv1d(7,7,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=7),
            nn.Conv1d(7,7,4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=7),
            nn.Conv1d(7,7,4),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, WPD):
        return self.sequence(WPD)

class DeepMC(pl.LightningModule):

    def __init__(self, seq_len = 24, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.loss = F.mse_loss
        self.seq_len = seq_len
        self.longscale = LSTMstack(self.seq_len)
        self.medium_1 = CNNstack(self.seq_len)
        self.medium_2 = CNNstack(self.seq_len)
        self.medium_3 = CNNstack(self.seq_len)
        self.shotscale = CNNstack(self.seq_len)
        self.attention_1 = Attention(self.seq_len,self.seq_len)
        self.attention_2 = Attention(self.seq_len,self.seq_len)

    def forward(self, batch):
        s1,m1,m2,m3,l1,y = batch
        h_j = self.longscale(l1)
        om1 = self.medium_1(m1)
        om2 = self.medium_1(m2)
        om3 = self.medium_1(m3)
        h_short = self.medium_1(s1)


        return om1 ,y

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self(batch)
        print(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer