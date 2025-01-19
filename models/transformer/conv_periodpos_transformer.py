import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, model_dim):
        super(PeriodicPositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.dims = [model_dim // 5] * 5  # 初始分配
        remainder = model_dim % 5  # 剩余的维度
        for i in range(remainder):  # 将剩余的维度均匀分配
            self.dims[i] += 1

    def forward(self, x, timestamps):
        """
        x: 输入张量，形状为 (batch_size, seq_len, model_dim)
        timestamps: 归一化的时间戳，形状为 (batch_size, seq_len, num_features)
                   num_features 对应季节、月份、小时、假期、星期几等
        """
        batch_size, seq_len, _ = timestamps.shape

        # 将 timestamps 重新映射到适合周期性编码的范围
        seasons = timestamps[:, :, 0] * 2 * math.pi  # 季节映射到 [0, 2π]
        months = timestamps[:, :, 1] * 2 * math.pi   # 月份映射到 [0, 2π]
        hours = timestamps[:, :, 2] * 2 * math.pi    # 小时映射到 [0, 2π]
        holidays = timestamps[:, :, 3]               # 假期（无需周期性编码）
        weekdays = timestamps[:, :, 4] * 2 * math.pi # 星期几映射到 [0, 2π]

        # 生成周期性编码
        def get_periodic_encoding(values, dim):
            """生成正弦和余弦编码"""
            positions = torch.arange(dim, dtype=torch.float, device=values.device)
            div_term = torch.exp(positions * (-math.log(10000.0) / dim))
            encoding = torch.zeros(batch_size, seq_len, dim, device=values.device)
            for i in range(dim // 2):
                encoding[..., 2 * i] = torch.sin(values * div_term[2 * i])
                encoding[..., 2 * i + 1] = torch.cos(values * div_term[2 * i + 1])
            return encoding

        # 对每个时间特征生成编码
        season_encoded = get_periodic_encoding(seasons, self.dims[0])  # 季节编码
        month_encoded = get_periodic_encoding(months, self.dims[1])    # 月份编码
        hour_encoded = get_periodic_encoding(hours, self.dims[2])      # 小时编码
        weekday_encoded = get_periodic_encoding(weekdays, self.dims[4])  # 星期几编码

        # 假期编码（直接使用线性映射）
        holiday_encoded = holidays.unsqueeze(-1).expand(-1, -1, self.dims[3])

        # 拼接所有编码
        periodic_encoding = torch.cat([
            season_encoded,
            month_encoded,
            hour_encoded,
            holiday_encoded,
            weekday_encoded
        ], dim=-1)  # (batch_size, seq_len, model_dim)

        # 将编码添加到输入中
        return x + periodic_encoding

class ConvTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_window, device, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        super(ConvTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_window = output_window
        self.device = device
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 周期性位置编码
        self.periodic_pe = PeriodicPositionalEncoding(model_dim)
        
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

        # 时间特征是第0-4
        src_timestamps = src[:, :, :5]  # (batch_size, seq_len, 5) -> [季节, 月份, 小时, 假期, 星期几]
        if tgt is not None:
            tgt_timestamps = tgt[:, :, :5]  

        # 嵌入输入
        src = self.embedding(src)  # (batch_size, seq_len, model_dim)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        
        # 应用周期性位置编码
        src = self.periodic_pe(src.permute(1, 0, 2), src_timestamps).permute(1, 0, 2)  # (seq_len, batch_size, model_dim)

        # 编码器
        memory = self.transformer_encoder(src)  # (seq_len, batch_size, model_dim)
        
        if tgt is not None:
            tgt = self.embedding(tgt)  # (batch_size, output_window, model_dim)
            tgt = tgt.permute(1, 0, 2)  # (output_window, batch_size, model_dim)
            
            # 应用周期性位置编码
            tgt = self.periodic_pe(tgt.permute(1, 0, 2), tgt_timestamps).permute(1, 0, 2)  # (output_window, batch_size, model_dim)

        # 初始化解码器输入
        decoder_input = src[-1:, :, :]  # 使用最后一个时间步作为初始输入
        outputs = []
        for t in range(self.output_window):
            # 生成自回归掩码
            tgt_mask = generate_autoregressive_mask(decoder_input.size(0)).to(self.device)
            
            # 解码器输出
            output_step = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)  # (seq_len, batch_size, model_dim)
            
            # 预测下一个时间步的值
            next_output = self.fc_out(output_step[-1:])  # (1, batch_size, 12)
            # print("next_output",next_output.shape)

            if tgt is not None:
                if torch.rand(1).item() < self.epsilon:
                    # 当 t < output_window - 1 时，使用目标序列的下一个时间步
                    next_input = tgt[t:t+1, :, :]  # (1, batch_size, model_dim)
                else:
                    # 随机数大于 self.epsilon 时，使用模型生成的输出
                    next_input = self.embedding(next_output)  # (1, batch_size, model_dim)
            else:
                # 如果没有目标序列，直接使用模型生成的输出
                next_input = self.embedding(next_output)  # (1, batch_size, model_dim)
            
            # 更新解码器输入
            decoder_input = torch.cat([decoder_input, next_input], dim=0)  # (seq_len + 1, batch_size, model_dim)
            
            # 保存当前时间步的输出
            outputs.append(next_output[-1, :, -1:])  # (batch_size, 1)

        # 将所有时间步的输出拼接起来
        output = torch.cat(outputs, dim=1)  # (batch_size, output_window)
        self.update_epsilon()
        return output