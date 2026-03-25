import torch
import torch.nn as nn
# from utils.time_embedding import timestep_embedding


def timestep_embedding(timesteps, dim):

    half_dim = dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32) / half_dim
    exponent = 10000 ** (-exponent)
    emb = timesteps.float().unsqueeze(1) * exponent.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb



class NoisePredictor(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim

        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # Output
        self.output = nn.Conv1d(64, 2, kernel_size=1)

    def forward(self, x, t):
        x = x.permute(0,2,1)
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(-1)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = h + t_emb
        h = self.act(self.conv3(h))
        out = self.output(h)
        return out.permute(0,2,1)