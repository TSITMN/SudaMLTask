import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_window, device , epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        """
        Transformer模型初始化。

        参数:
        - input_dim (int): 输入特征的维度。
        - model_dim (int): 模型的隐藏层维度(即Transformer中每个时间步的向量维度)。
        - num_heads (int): 多头注意力机制中的头数。
        - num_encoder_layers (int): 编码器(Encoder)的层数。
        - num_decoder_layers (int): 解码器(Decoder)的层数。
        - output_window (int): 输出序列的长度(例如,预测未来96小时的数据)。
        - device: 使用的设备（如 'cuda' 或 'cpu'）。
        """
        super(Transformer, self).__init__()

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # self.device = device
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_window = output_window
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, output_window, model_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads , dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads , dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        # 输出层
        self.fc_out = nn.Linear(model_dim, 12)  # 每次预测一个时间步的值

    def update_epsilon(self):
        """Update epsilon for scheduled sampling."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def forward(self, src, tgt=None , scaler=None):
        """
        前向传播。

        参数:
        - src: 编码器输入，形状为 (batch_size, seq_len, input_dim)。
        - tgt: 解码器输入（目标序列），形状为 (batch_size, output_window, input_dim)。
        - scheduled_sampling_prob: 使用真实目标序列的概率（Scheduled Sampling）。
        """
        batch_size = src.size(0)

        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        if tgt is not None:
            tgt = tgt.permute(1, 0, 2)  
        # 初始化解码器输入
        decoder_input = src[-2:-1, :, :]  # 初始输入是原始序列的最后时间步，形状为 (1, batch_size, model_dim)
        outputs = []
        for t in range(self.output_window):
            # 解码器输出
            output_step = self.transformer_decoder(decoder_input, memory)  
            # 预测下一个时间步的值
            next_output = self.fc_out(output_step[-1::])  # (1 , batch_size , 12 )

            if tgt is not None:
                if torch.rand(1).item() < self.epsilon :
                    # 当 t < output_window - 1 时，使用目标序列的下一个时间步
                    next_input = tgt[t:t+1, :, :]  
                else:
                    # 随机数大于 self.epsilon 时，使用模型生成的输出
                    next_input = next_output 
            else:
                next_input = next_output
                
            next_input = self.embedding(next_input) #(1, batch_size, model_dim)
            
            # 更新解码器输入
            decoder_input = torch.cat([decoder_input, next_input], dim=0)  # (seq_len + 1, batch_size, model_dim)
            
            # 保存当前时间步的输出
            outputs.append(next_output[-1,:,-1:]) # (batch_size, 1)

        # 将所有时间步的输出拼接起来
        output = torch.cat(outputs, dim=1)  # (batch_size, output_window)
            
        self.update_epsilon()
        return output
        