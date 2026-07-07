import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1    = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2      = nn.BatchNorm1d(out_ch)

    def forward(self, x, t_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.relu(self.bn2(self.conv2(h)))
        return h
