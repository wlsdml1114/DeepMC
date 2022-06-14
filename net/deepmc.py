import torch
import copy

import torch.nn.functional as F
import pytorch_lightning as pl

from net.encoder import CNNstack, LSTMstack
from net.attention import Position_based_content_attention, Scaled_Guided_Attention
from net.decoder import Decoder

class DeepMC(pl.LightningModule):

    def __init__(self,num_encoder_hidden : int , num_encoder_times : int, num_decoder_hidden : int,num_decoder_times : int, batch_size : int, num_of_CNN_stacks : int, cnn_output_size : int, num_feature : int, seq_len : int = 24, lr : int = 1e-3):
        super().__init__()

        # Hyper parameter
        self.num_encoder_hidden = num_encoder_hidden
        self.num_encoder_times = num_encoder_times
        self.num_decoder_hidden = num_decoder_hidden
        self.num_decoder_times = num_decoder_times
        self.batch_size = batch_size
        self.num_of_CNN_stacks = num_of_CNN_stacks
        self.cnn_output_size = cnn_output_size
        self.num_feature = num_feature
        # num_feature = Z
        # X = {Z, Y, {Rlat, Rlong}}
        # encoder input = (X,U)
        # thats why num_feature + 4 = num_encoder_feature
        self.num_encoder_feature = self.num_feature + 3 + 1
        self.lr = lr
        self.loss = F.mse_loss
        self.seq_len = seq_len

        # Encoder 
        # LSTMstack for long scale
        self.LSTMstack = LSTMstack(self.num_encoder_feature).to(self.device)

        # CNNstacks for middel & short scale
        cnnstack = CNNstack(self.num_encoder_feature).to(self.device)
        self.CNNstacks = [copy.deepcopy(cnnstack) for i in range(self.num_of_CNN_stacks)]
        
        # set where CNN should be look
        # 0 is shortest scale, 5 is longest scale
        self.X_levels = [0,0,1,2,1,2,3]
        self.U_levels = [0,1,0,1,2,3,2]

        # Attention layer
        # Position_based_content_attention
        # input / CNNLSTM encoder output & s_i-1
        # output / Position_based_content_attention context vector c_i
        self.Position_based_content_attention = Position_based_content_attention(
            num_encoder_hidden=self.num_encoder_hidden, 
            num_encoder_times=self.num_encoder_times,
            num_decoder_hidden = self.num_decoder_hidden,
            num_decoder_times= self.num_decoder_times, 
            batch_size= self.batch_size
        ).to(self.device)

        # Scaled_Guided_Attention
        # intput / CNN encoder output & s_i-1
        # output / Scaled_Guided_Attention context vector c_prime_i
        self.Scaled_Guided_Attention = Scaled_Guided_Attention(
            num_encoder_hidden=self.num_encoder_hidden, 
            num_encoder_times=self.num_encoder_times,
            num_decoder_hidden = self.num_decoder_hidden,
            num_decoder_times= self.num_decoder_times, 
            batch_size= self.batch_size, 
            num_of_CNN_stacks = self.num_of_CNN_stacks, 
            cnn_output_size = self.cnn_output_size
        ).to(self.device)

        # Decoder
        # hidden state / (batch size, decoder hidden size)
        self.s_i = torch.rand((self.batch_size,self.num_decoder_hidden))
        # cell state / (batch size, decoder hidden size)
        self.cell_state = torch.rand((self.batch_size,self.num_decoder_hidden))
        # decoder output / (batch size, 1)
        self.m_i = torch.rand((self.batch_size, 1))

        # Decoder layer
        # input / attention context vector c_i + c_prime_i
        # output / lstm decoder output
        self.Decoder = Decoder(
            num_encoder_hidden=self.num_encoder_hidden, 
            num_decoder_hidden = self.num_decoder_hidden,
            cnn_output_size = self.cnn_output_size
        ).to(self.device)

    def forward(self, batch):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12
        X = batch[0]
        U = batch[1]

        # Encoder
        # LSTM looks long scale 
        # thats why X & U level is 4
        LSTM = self.LSTMstack(torch.cat((X[:,4,:,:],U[:,4,:,:]),1))

        # CNN looks middle & short scale
        CNNs = [
            self.CNNstacks[i](torch.cat((X[:,self.X_levels[i],:,:],U[:,self.U_levels[i],:,:]),1)) for i in range(self.num_of_CNN_stacks)
        ]
        CNNs = torch.stack(CNNs,dim=1)

        # Attention & decoder
        output = []
        for i in range(self.num_decoder_times):

            # c_i / (batch size, 1, encoder hidden size)
            c_i = self.Position_based_content_attention(LSTM, self.s_i, i)

            # CNNs / (batch size, # of CNN stack, output size of CNN stack)
            # c_prime_i / (batch size, 1, output size of CNN stack)
            c_prime_i = self.Scaled_Guided_Attention(CNNs, self.s_i)
            
            # Decoder
            # decoder_input / (batch size, 1(output of decoder) + encoder hidden size * 2 + output size of CNN stack)
            decoder_input = torch.cat((self.m_i, c_i.squeeze(1),c_prime_i.squeeze(1)),dim=1)
            
            # s_i / hidden state of Decoder LSTM / (batch size, decoder hidden size)
            # cell_state / cell state of Decoder LSTM / (batch size, decoder hidden size)
            # m_i / outptu of decoder / (batch size, 1)
            self.m_i, (self.s_i, self.cell_state) = self.Decoder(decoder_input,self.s_i,self.cell_state)

            output.append(self.m_i)

        # output (decoder timestep, batch size, 1)
        # after permute output (batch size, decoder timestep, 1)
        output = torch.stack(output, dim=0).permute(1,0,2)
        
        return output

    def training_step(self, batch, batch_idx):
        X = batch[0]
        U = batch[1]
        Target = batch[2]
        y_hat = self([X,U])

        loss = self.loss(y_hat, Target.unsqueeze(2))
        print(loss)
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.manual_backward(loss, retain_graph=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        U = batch[1]
        Target = batch[2]
        y_hat = self([X,U])

        loss = self.loss(y_hat, Target.unsqueeze(2))
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        X = batch[0]
        U = batch[1]
        Target = batch[2]
        y_hat = self([X,U])

        loss = self.loss(y_hat, Target)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        X = batch[0]
        U = batch[1]
        Target = batch[2]
        y_hat = self([X,U])

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @property
    def automatic_optimization(self) -> bool:
        return False