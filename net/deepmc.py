import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import dropout, nn


class Attention_Decoder(nn.Module):
    def __init__(self,num_encoder_hidden : int , num_encoder_times : int, num_decoder_hidden : int,num_decoder_times : int, batch_size : int = 1):
        super().__init__()
        self.num_encoder_hidden = num_encoder_hidden
        self.num_encoder_times = num_encoder_times
        self.num_decoder_hidden = num_decoder_hidden
        self.num_decoder_times = num_decoder_times
        self.batch_size = batch_size

        # in this case 7
        self.num_of_CNN_stacks = 7
        self.cnn_output_size = 105

        self.W_a = nn.Linear(self.num_decoder_hidden, self.num_decoder_times)
        self.U_a = nn.Linear(self.num_encoder_hidden, self.num_encoder_times)
        self.v_a = nn.Linear(self.num_encoder_times+self.num_decoder_times,1)
        self.phi_weight = nn.Linear(self.num_encoder_times, self.num_encoder_hidden)

        # hidden state / (batch size, decoder hidden size)
        self.s_i = torch.rand((self.batch_size,self.num_decoder_hidden))
        # cell state / (batch size, decoder hidden size)
        self.m_i = torch.rand((self.batch_size,self.num_decoder_hidden))

        # m_i = G(m_i-1, s_i-1, c_i)
        # m_i-1 / (num_decoder_hidden)
        # s_i-1 / (num_decoder_hidden)
        # c_i   / (num_of_CNN_stacks + num_encoder_hidden) 
        self.Decoder = nn.LSTMCell(self.cnn_output_size+self.num_encoder_hidden, self.num_decoder_hidden)
        
        # 105 is output size of CNN stack
        self.w_i_j_T = [nn.Linear(self.num_decoder_hidden+self.cnn_output_size,1) for i in range(self.num_of_CNN_stacks)]
    
    def forward(self, LSTM, CNNs):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12


        # Position based content attention layer
        output = []
        for i in range(self.num_decoder_times):

            
            v_a_output = []
            for j in range(self.num_encoder_times):
                    
                # delta_i_j / (encoder time step)
                delta_i_j = torch.zeros((self.num_encoder_times))
                delta_i_j[:j] = 1

                # phi_delta / (encoder hidden size)
                phi_delta = self.phi_weight(delta_i_j)

                # LSTM / (batch size, encoder time step, encoder hidden size)
                # phi_delta_hadamard / (batch size, encoder hidden size)
                phi_delta_hadamard = phi_delta*LSTM[:,j]

                # U_a_output / (batch size, encoder time step)
                U_a_output = self.U_a(phi_delta_hadamard)
                
                # W_a_output / (batch size, decoder time step)
                W_a_output = self.W_a(self.s_i)

                # concat / (batch,size, decoder time step + encoder time step)
                concat = torch.cat((W_a_output,U_a_output), dim=1)

                # delta_i_t_j / (decoder time step + encoder time step)
                delta_i_T_j = torch.zeros((self.num_encoder_times+self.num_decoder_times))
                delta_i_T_j[i+self.num_encoder_times-j] = 1

                # concat_tanh / (batch size, decoder time step + encoder time step)
                concat_tanh = torch.tanh(concat)

                # concat_tanh_hadamard_delta / (batch size, decoder time step + encoder time step)
                concat_tanh_hadamard_delta = concat_tanh*delta_i_T_j

                # v_a_output / (batch size, 1)
                v_a_output.append(self.v_a(concat_tanh_hadamard_delta))
            
            # e_ij / (batch size, encoder time step)
            e_ij = torch.cat(v_a_output,dim=1)

            # a_ij / (batch size, encoder time step)
            a_ij = torch.softmax(e_ij,dim=1)

            # a_ij / (batch size, 1, encoder time step)
            a_ij = a_ij.unsqueeze(1)

            # LSTM / (batch size, encoder time step, encoder hidden size)
            # c_i / (batch size, 1, encoder hidden size)
            c_i = torch.bmm(a_ij,LSTM)

            # Scaled Guided Attention Layer
            e_prime_ij = []
            for k in range(self.num_of_CNN_stacks):
                # CNNs / (batch size, # of CNN stack, output size of CNN stack)
                # s_i / (batch size, decoder hidden size)
                e_prime_ij.append(self.w_i_j_T[k](torch.cat((CNNs[:,k,:],self.s_i),dim=1)))
            
            # e_prime_ij / (batch size, # of CNN stack)
            e_prime_ij = torch.tanh(torch.cat(e_prime_ij,dim=1))

            # a_prime_ij / (batch size, # of CNN stack)
            # attention is not softmax in Scaled Guided Attention Layer
            a_prime_ij = torch.exp(e_prime_ij)/torch.sum(e_prime_ij)

            # a_prime_ij / (batch size, 1, # of CNN stack)
            a_prime_ij = a_prime_ij.unsqueeze(1)

            # CNNs / (batch size, # of CNN stack, output size of CNN stack)
            # c_prime_i / (batch size, 1, output size of CNN stack)
            c_prime_i = torch.bmm(a_prime_ij,CNNs)
            
            # Decoder
            # decoder_input / (batch size, encoder hidden size + output size of CNN stack)
            decoder_input = torch.cat((c_i.squeeze(1),c_prime_i.squeeze(1)),dim=1)

            # hx / hidden state of Decoder LSTM / (batch size, decoder hidden size)
            # cx / cell state of Decoder LSTM / (batch size, decoder hidden size)
            hx, cx = self.Decoder(decoder_input,(self.s_i,self.m_i))

            output.append(hx)

        # output (decoder timestep, batch size, decoder hidden size)
        # after permute output (batch size, decoder timestep, decoder hidden size)
        output = torch.stack(output, dim=0).permute(1,0,2)
        
        return output

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