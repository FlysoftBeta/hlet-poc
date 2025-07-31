import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, in_size: int, imm_size: int, out_size: int):
        super().__init__()
        self.up = nn.Linear(in_size, imm_size)
        self.act = nn.SiLU()
        self.down = nn.Linear(imm_size, out_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        return self.down(self.dropout(self.act(self.up(x))))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, imm_size: int, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=0.2, batch_first=True
        )
        self.ffn = FFN(hidden_size, imm_size, hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):

        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_base_layers = 4
        self.base_size = 2048
        self.base_imm_size = 3072
        self.num_decoder_layers = 4
        self.hidden_size = 256
        self.imm_size = 512
        self.max_value = 100

        self.base_layers = nn.ModuleList(
            [
                DecoderLayer(self.base_size, self.base_imm_size)
                for _ in range(self.num_base_layers)
            ]
        )
        self.base_proj = nn.Linear(self.base_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(self.hidden_size, self.imm_size)
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.norm = nn.RMSNorm(self.hidden_size)
        self.num_pred_head = nn.Linear(self.hidden_size, self.max_value)

    def forward(
        self,
        base_states: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        base_layers_out = base_states
        for layer in self.base_layers:
            base_layers_out = layer(base_layers_out, attn_mask)
        base_proj_out = self.base_proj(base_states)
        decoder_out = base_proj_out
        for layer in self.decoder_layers:
            decoder_out = layer(decoder_out, attn_mask)

        pred_result = self.num_pred_head(self.norm(decoder_out))
        return pred_result[:,-1,:]

__all__ = [Model]
