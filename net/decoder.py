from torch import nn

class Decoder(nn.Module):
    def __init__(self,num_encoder_hidden : int , num_decoder_hidden : int, cnn_output_size : int):
        
        super().__init__()
        
        # Hyper parameter
        self.num_encoder_hidden = num_encoder_hidden
        self.num_decoder_hidden = num_decoder_hidden
        self.cnn_output_size = cnn_output_size
        
        # m_i = G(m_i-1, s_i-1, c_i)
        # m_i-1 / (1)
        # s_i-1 / (num_decoder_hidden)
        # c_i   / (num_of_CNN_stacks + num_encoder_hidden * 2) 
        self.Decoder = nn.LSTMCell(1+self.cnn_output_size+self.num_encoder_hidden * 2, self.num_decoder_hidden)

        # FC Layer for hidden state to output
        # 2 layers / each layer has 50, 1 output dimension
        self.FC_layer = nn.Sequential(
            nn.Linear(self.num_decoder_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, decoder_input, s_i, cell_state):
        # 1 <= j <= T  , num_encoder_times T is lstmstack seq length, in this case T = 18
        # 1 <= i <= T' , num_decoder_times T' is lstm decoder seq length, in this case T' = 12
    
        # decoder_input / input of decoder / (m_i-1, (s_i-1, cell_state), (c_i+c_prime_i))
        # m_i / output of decoder / (batch size, 1)
        # s_i-1 / hidden state of Decoder LSTM / (batch size, decoder hidden size)
        # cell_state / cell state of Decoder LSTM / (batch size, decoder hidden size)
        # c_i / (batch size, 1, encoder hidden size)
        # c_prime_i / (batch size, 1, output size of CNN stack)
        s_i, cell_state = self.Decoder(decoder_input,(s_i,cell_state))

        # m_i / output of decoder / (batch size, 1)
        m_i = self.FC_layer(s_i)
        
        return m_i,(s_i,cell_state)