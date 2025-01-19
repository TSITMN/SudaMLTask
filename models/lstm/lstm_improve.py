import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size, scale_factor=4):
        super().__init__()
        """
        Attention module of the LSTM model.
        
        Args:
            hidden_size (int): The number of features in the hidden state h.
        """
        self.scale_factor = scale_factor
        self.encoder_projection = nn.Linear(hidden_size*self.scale_factor, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size*self.scale_factor, hidden_size)
        
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(1, hidden_size))  # Shape changed to (1, hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        hidden = self.hidden_projection(hidden[-1])  # (batch_size, hidden_size)
        encoder_outputs = self.encoder_projection(encoder_outputs)
        
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_size)
        energy = energy.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
        
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 注意力权重归一化
        
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)
        
        return context_vector, attn_weights

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

        return output, h, c
    
class DecoderWithAttention(nn.Module):
    def __init__(self, input_feature_len, hidden_size=512, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scale_factor = 8

        self.lstm = nn.LSTM(input_size=self.input_feature_len,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.attention_hidden_size = hidden_size // self.scale_factor
        self.attention = Attention(self.attention_hidden_size, self.scale_factor)
        self.linear = nn.Linear(in_features=self.hidden_size + self.attention_hidden_size, out_features=self.input_feature_len)

    def forward(self, input_seq, h, c, encoder_outputs):
        input_seq = input_seq.unsqueeze(1) # (batch_size, 1, input_size)
        context_vector, attn_weights = self.attention(h, encoder_outputs) # (batch_size, att_hidden_size)
        lstm_out, (h, c) = self.lstm(input_seq, (h, c)) # output(batch_size, 1, hidden_size)
        
        output = torch.cat((lstm_out.squeeze(1), context_vector), dim=1) # (batch_size, hidden_size + att_hidden_size)
        pred = self.linear(output)
        return pred, h, c, attn_weights

    
class Seq2seqLSTMWithCNN(nn.Module):
    def __init__(self, input_feature_len, hidden_size, num_layers, output_size, device='cuda', epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, kernel_size=3):
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
            kernel_size (int): The size of the kernel.
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
        
        self.conv = nn.Conv1d(
            in_channels=input_feature_len,
            out_channels=input_feature_len,  # 保持相同的特征维度
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
    def update_epsilon(self):
        """Update epsilon for scheduled sampling."""
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def forward(self, input_seq, target_seq=None):
        target_len = self.output_size  # The length of the output sequence
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        
        # CNN
        input_seq = input_seq.permute(0, 2, 1) # (batch_size, input_feature_len, seq_len)
        input_seq = self.conv(input_seq)
        input_seq = input_seq.permute(0, 2, 1)
        
        encode_output, h, c = self.Encoder(input_seq)
        
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
    
class Seq2seqLSTMWithAttention(nn.Module):
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
        self.Decoder = DecoderWithAttention(self.input_feature_len, self.hidden_size, self.num_layers)
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
        
        encode_output, h, c = self.Encoder(input_seq)
        
        outputs = torch.zeros(batch_size, self.input_feature_len, self.output_size).to(self.device)
        decoder_input = input_seq[:, -1, :]
        
        for t in range(target_len):
            decoder_output, h, c, _ = self.Decoder(decoder_input, h, c, encode_output)
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
    
class Seq2seqLSTMWithAttentionCNN(nn.Module):
    def __init__(self, input_feature_len, hidden_size, num_layers, output_size, device='cuda', epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, kernel_size=3):
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
            kernel_size (int): The size of the kernel.
        """
        
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        self.Encoder = Encoder(self.input_feature_len, self.hidden_size, self.num_layers)
        self.Decoder = DecoderWithAttention(self.input_feature_len, self.hidden_size, self.num_layers)
        
        # Parameters of the scheduled sampling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training = True
        
        self.conv = nn.Conv1d(
            in_channels=input_feature_len,
            out_channels=input_feature_len,  # 保持相同的特征维度
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
    def update_epsilon(self):
        """Update epsilon for scheduled sampling."""
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def forward(self, input_seq, target_seq=None):
        target_len = self.output_size  # The length of the output sequence
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        
        # CNN
        input_seq = input_seq.permute(0, 2, 1)
        input_seq = self.conv(input_seq)
        input_seq = input_seq.permute(0, 2, 1)
        
        encode_output, h, c = self.Encoder(input_seq)
        
        outputs = torch.zeros(batch_size, self.input_feature_len, self.output_size).to(self.device)
        decoder_input = input_seq[:, -1, :]
        
        for t in range(target_len):
            decoder_output, h, c, _ = self.Decoder(decoder_input, h, c, encode_output)
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