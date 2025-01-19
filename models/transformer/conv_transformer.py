import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_autoregressive_mask(seq_len):
    """
    生成自回归掩码。
    返回形状为 (seq_len, seq_len) 的下三角矩阵。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class Conv1DBlock(nn.Module):
    """
    1D 卷积块，用于捕捉局部特征。
    """
    def __init__(self, model_dim, kernel_size=3, dropout=0.1):
        super(Conv1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(model_dim, model_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (seq_len, batch_size, model_dim)
        """
        x = x.permute(1, 2, 0)  # (batch_size, model_dim, seq_len)
        x = self.conv1(x)  # (batch_size, model_dim, seq_len)
        x = F.relu(x)
        x = self.conv2(x)  # (batch_size, model_dim, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, model_dim)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class ConvTransformerEncoderLayer(nn.Module):
    """
    卷积 Transformer 编码器层。
    """
    def __init__(self, model_dim, num_heads, kernel_size=3, dropout=0.1):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.conv_block = Conv1DBlock(model_dim, kernel_size, dropout)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False , memory_is_causal=False , tgt_is_causal=False):
        """
        src: (seq_len, batch_size, model_dim)
        src_mask: (seq_len, seq_len)
        src_key_padding_mask: (batch_size, seq_len)
        """
        # 卷积块
        src2 = self.conv_block(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm2(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        return src

class ConvTransformerDecoderLayer(nn.Module):
    """
    卷积 Transformer 解码器层。
    """
    def __init__(self, model_dim, num_heads, kernel_size=3, dropout=0.1):
        super(ConvTransformerDecoderLayer, self).__init__()
        self.conv_block = Conv1DBlock(model_dim, kernel_size, dropout)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                is_causal=False , memory_is_causal=False , tgt_is_causal=False):
        """
        tgt: (seq_len, batch_size, model_dim)
        memory: (seq_len, batch_size, model_dim)
        tgt_mask: (seq_len, seq_len)
        memory_mask: (seq_len, seq_len)
        tgt_key_padding_mask: (batch_size, seq_len)
        memory_key_padding_mask: (batch_size, seq_len)
        """
        # 卷积块
        tgt2 = self.conv_block(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # 自注意力（带自回归掩码）
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # 编码器-解码器注意力
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_window, device, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_window = output_window
        self.device = device
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 编码器和解码器
        self.encoder_layer = ConvTransformerEncoderLayer(model_dim, num_heads, kernel_size=3, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_layer = ConvTransformerDecoderLayer(model_dim, num_heads, kernel_size=3, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        # 输出层
        self.fc_out = nn.Linear(model_dim, 12)  # 每次预测一个时间步的值

    def update_epsilon(self):
        """Update epsilon for scheduled sampling."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def forward(self, src, tgt=None):
        """
        前向传播。

        参数:
        - src: 编码器输入，形状为 (batch_size, seq_len, input_dim)。
        - tgt: 解码器输入（目标序列），形状为 (batch_size, output_window, input_dim)。
        """
        batch_size = src.size(0)

        # 嵌入输入
        src = self.embedding(src)  # (batch_size, seq_len, model_dim)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        
        # 编码器
        memory = self.transformer_encoder(src)  # (seq_len, batch_size, model_dim)
        
        if tgt is not None:
            # tgt = self.embedding(tgt)  # (batch_size, output_window, model_dim)
            tgt = tgt.permute(1, 0, 2)  # (output_window, batch_size, input_dim)
        
        # 初始化解码器输入
        decoder_input = src[-1:, :, :]  # 使用最后一个时间步作为初始输入
        outputs = []
        for t in range(self.output_window):
            # 生成自回归掩码
            tgt_mask = generate_autoregressive_mask(decoder_input.size(0)).to(self.device)
            
            # 解码器输出
            output_step = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)  # (seq_len, batch_size, model_dim)
            
            # 预测下一个时间步的值
            next_output = self.fc_out(output_step[-1:])  # (1, batch_size, 1)

            if tgt is not None:
                if torch.rand(1).item() < self.epsilon :
                    # 当 t < output_window - 1 时，使用目标序列的下一个时间步
                    next_input = tgt[t:t+1, :, :]  
                else:
                    # 随机数大于 self.epsilon 时，使用模型生成的输出
                    next_input = next_output 
            else:
                next_input = next_output
            
            # 嵌入下一个输入
            next_input = self.embedding(next_input)  # (1, batch_size, model_dim)
            
            # 更新解码器输入
            decoder_input = torch.cat([decoder_input, next_input], dim=0)  # (seq_len + 1, batch_size, model_dim)
            
            # 保存当前时间步的输出
            outputs.append(next_output[-1, :, -1:])  # (batch_size, 1)

        # 将所有时间步的输出拼接起来
        output = torch.cat(outputs, dim=1)  # (batch_size, output_window)
        self.update_epsilon()
        return output