import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size=512, num_layers=1, device='cuda'):
        super().__init__()
        """
        Encoder module of the LSTM model.
        
        Args:
            input_feature_len (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers
        """
        
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.device = device
        self.lstm = nn.LSTM(
            input_size=self.input_feature_len, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=False
        )
        
    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c
    
class Decoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size=512, num_layers=1):
        super().__init__()
        """
        Decoder module of the LSTM model.
        
        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
        """
        
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.lstm = nn.LSTM(
            input_size=self.input_feature_len, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.input_feature_len)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, seq_len, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num_directions * hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, input_feature_len)

        return pred, h, c
    
class Seq2seqLSTM(nn.Module):
    def __init__(self, input_feature_len, hidden_size, num_layers, output_size, device='cuda', epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        super().__init__()
        """
        Encoder-Decoder LSTM model. 
        
        Args:
            input_feature_len (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            output_size (int): The number of expected features in the output y.
            epsilon_start (float): The initial value of scheduled sampling.
            epsilon_end (float): The final value of scheduled sampling.
            epsilon_decay (int): The number of iterations to decay epsilon from epsilon_start to epsilon_end.
        """
        
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        self.Encoder = Encoder(self.input_feature_len, self.hidden_size, self.num_layers)
        self.Decoder = Decoder(self.input_feature_len, self.hidden_size, self.num_layers)
        
        # Parameters of the scheduled sampling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training = True
        
    def update_epsilon(self):
        """Update epsilon for scheduled sampling."""
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def forward(self, input_seq, target_seq=None):
        target_len = self.output_size  # The length of the output sequence
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        
        h, c = self.Encoder(input_seq)
        
        outputs = torch.zeros(batch_size, self.input_feature_len, self.output_size).to(self.device)
        decoder_input = input_seq[:, -1, :]
        
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = decoder_output
            
            # Scheduled sampling
            if self.training and target_seq is not None:
                if torch.rand(1) < self.epsilon:
                    decoder_input = target_seq[:, t, :]
                else:
                    decoder_input = decoder_output
            else:
                decoder_input = decoder_output

        self.update_epsilon()
        return outputs[:, -1, :]