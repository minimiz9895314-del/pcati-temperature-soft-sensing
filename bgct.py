import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.feature_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        Hf = torch.tanh(self.feature_conv(x))
        Hg = torch.sigmoid(self.gate_conv(x))
        return Hg * Hf

class GatedConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcu3 = GatedConvUnit(in_channels, out_channels, 3)
        self.gcu5 = GatedConvUnit(in_channels, out_channels, 5)
        self.gcu7 = GatedConvUnit(in_channels, out_channels, 7)
        self.res_proj = nn.Conv1d(in_channels, 3 * out_channels, 1)

    def forward(self, x):
        H3 = self.gcu3(x)
        H5 = self.gcu5(x)
        H7 = self.gcu7(x)
        H_gcn = torch.cat([H3, H5, H7], dim=1)
        return H_gcn + self.res_proj(x)

class TemperatureAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.tau = nn.Parameter(torch.tensor(10.0))  # Thermal time constant

    def forward(self, x, pos_enc):
        batch_size, seq_len, d_model = x.shape
        Q = self.q_linear(x + pos_enc)
        K = self.k_linear(x + pos_enc)
        V = self.v_linear(x + pos_enc)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Temperature-aware mask
        i = torch.arange(seq_len).unsqueeze(0).unsqueeze(2)
        j = torch.arange(seq_len).unsqueeze(0).unsqueeze(1)
        mask = self.beta * torch.exp(-self.gamma * torch.abs(i - j) / self.tau)
        scores = scores + mask.to(scores.device)
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(context)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.attn = TemperatureAwareAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pos_enc):
        attn_out = self.attn(self.norm1(x), pos_enc)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        return x + ffn_out

class BayesianGatedConvTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, num_heads=4, dim_ff=256, num_layers=2, dropout_rates=[0.2, 0.4]):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.gcm = GatedConvModule(d_model, d_model // 3)
        self.encoders = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.highway = nn.Linear(d_model, d_model)  # Highway network as residual
        self.mean_head = nn.Linear(d_model, 1)
        self.var_head = nn.Linear(d_model, 1)
        self.pos_enc = self._generate_pos_enc(1000, d_model)  # Precompute positional encodings
        self.conv_dropout = nn.Dropout(dropout_rates[0])
        self.fc_dropout = nn.Dropout(dropout_rates[1])

    def _generate_pos_enc(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x, training=True):
        if training:
            x = self.conv_dropout(x)
        x = self.embedding(x)
        pos_enc = self.pos_enc[:x.shape[1], :].to(x.device).unsqueeze(0)
        x = self.gcm(x.transpose(1, 2)).transpose(1, 2)  # Conv expects [B, C, L]
        for encoder in self.encoders:
            x = encoder(x, pos_enc)
        x = x + self.highway(x)  # Highway residual
        if training:
            x = self.fc_dropout(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        mean = self.mean_head(x)
        var = F.softplus(self.var_head(x))  # Ensure positive variance
        return mean, var