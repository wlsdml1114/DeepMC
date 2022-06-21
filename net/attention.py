import torch
import copy 

from torch import nn

class Position_based_content_attention(nn.Module):
    # input / CNNLSTM encoder output & s_i-1
    # output / Position_based_content_attention context vector c_i
    def __init__(self,num_encoder_hidden : int , num_encoder_times : int, num_decoder_hidden : int,num_decoder_times : int, batch_size : int):
        
        super().__init__()

        # Hyper parameter
        self.num_encoder_hidden = num_encoder_hidden
        self.num_encoder_times = num_encoder_times
        self.num_decoder_hidden = num_decoder_hidden
        self.num_decoder_times = num_decoder_times
        self.batch_size = batch_size

        # Weights
        self.W_a = nn.Linear(self.num_decoder_hidden, self.num_decoder_times)
        self.U_a = nn.Linear(self.num_encoder_hidden*2, self.num_encoder_times)
        self.v_a = nn.Linear(self.num_encoder_times+self.num_decoder_times,1)
        self.phi_weight = nn.Linear(self.num_encoder_times + self.num_decoder_times, self.num_encoder_hidden * 2)
    
    def forward(self, LSTM, s_i, i):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12
        # Position based content attention layer
        v_a_output = []
        for j in range(self.num_encoder_times):
                
            # delta_i_j / (encoder time step + decoder time step)
            delta_i_j = torch.zeros((self.num_encoder_times + self.num_decoder_times)).cuda()
            delta_i_j[i+self.num_encoder_times-j] = 1

            # phi_delta / (encoder hidden size * 2)
            phi_delta = self.phi_weight(delta_i_j)

            # LSTM / (batch size, encoder time step, encoder hidden size * 2)
            # phi_delta_hadamard / (batch size, encoder hidden size * 2)
            phi_delta_hadamard = phi_delta*LSTM[:,j]

            # U_a_output / (batch size, encoder time step)
            U_a_output = self.U_a(phi_delta_hadamard)
            
            # W_a_output / (batch size, decoder time step)
            W_a_output = self.W_a(s_i)

            # concat / (batch,size, decoder time step + encoder time step)
            concat = torch.cat((W_a_output,U_a_output), dim=1)

            
            # delta_i_t_j / (decoder time step + encoder time step)
            delta_i_T_j = torch.zeros((self.num_encoder_times+self.num_decoder_times)).cuda()
            delta_i_T_j[:self.num_encoder_times] = 1

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

        # LSTM / (batch size, encoder time step, encoder hidden size * 2)
        # c_i / (batch size, 1, encoder hidden size)
        c_i = torch.bmm(a_ij,LSTM)

        return c_i

class Scaled_Guided_Attention(nn.Module):
    # intput / CNN encoder output & s_i-1
    # output / Scaled_Guided_Attention context vector c_prime_i
    def __init__(self,num_encoder_hidden : int , num_encoder_times : int, num_decoder_hidden : int,num_decoder_times : int, batch_size : int, num_of_CNN_stacks : int, cnn_output_size : int):
        
        super().__init__()
        
        # Hyper parameter
        self.num_encoder_hidden = num_encoder_hidden
        self.num_encoder_times = num_encoder_times
        self.num_decoder_hidden = num_decoder_hidden
        self.num_decoder_times = num_decoder_times
        self.batch_size = batch_size
        self.num_of_CNN_stacks = num_of_CNN_stacks
        self.cnn_output_size = cnn_output_size
        
        # energy weight
        #linear = nn.Linear(self.num_decoder_hidden+self.cnn_output_size,1)
        self.w_i_j_T = [nn.Linear(self.num_decoder_hidden+self.cnn_output_size,1)
                        for _ in range(self.num_of_CNN_stacks)]
        self.w_i_j_T = torch.nn.ModuleList(self.w_i_j_T)
    
    def forward(self, CNNs, s_i):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12

        # energy vector
        e_prime_ij = []

        for i in range(self.num_of_CNN_stacks):
            # CNNs / (batch size, # of CNN stack, output size of CNN stack)
            # s_i / (batch size, decoder hidden size)
            e_prime_ij.append(self.w_i_j_T[i](torch.cat((CNNs[:,i,:],s_i),dim=1)))
        
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
        
        return c_prime_i