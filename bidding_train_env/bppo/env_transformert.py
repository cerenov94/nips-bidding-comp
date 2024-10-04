import torch
import torch.nn as nn
from torch.distributions import Normal


class ENVT(nn.Module):
    def __init__(self, state_dim=16, action_dim=1, dim=192, depth=6, head_size=32, max_episode_length=48):
        super().__init__()

        self.timestep_emb = nn.Embedding(max_episode_length, dim)
        self.action_emb = nn.Linear(action_dim, dim // 4)
        self.state_emb = nn.Linear(state_dim, dim // 4 * 3)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim // head_size, dim_feedforward=4 * dim, dropout=0.1,
                                       activation=nn.ReLU(), batch_first=True, norm_first=True),
            num_layers=depth
        )
        self.state_mu = nn.Linear(dim, state_dim + action_dim)

        self.state_std = nn.Linear(dim, state_dim + action_dim)

        self.noise_std = nn.Linear(dim, state_dim + action_dim)

    def forward(self, state, action, timestep, mask):
        state = self.state_emb(state)
        action = self.action_emb(action)
        sa = torch.cat([state, action], dim=-1)
        timestep = self.timestep_emb(timestep)

        sa = sa + timestep

        sa = self.transformer(sa, src_key_padding_mask=~mask)

        state_mu = self.state_mu(sa)
        state_log_std = self.state_std(sa)
        state_log_std = torch.clamp(state_log_std, -20, 2)
        state_std = state_log_std.exp()

        noised_log_std = self.noise_std(sa)
        noised_log_std = torch.clamp(noised_log_std, -20, 2)
        noise_std = noised_log_std.exp()

        state_pdf = Normal(state_mu, state_std)
        noise_pdf = Normal(torch.zeros_like(noise_std).to(noise_std.device), noise_std)

        return state_pdf, noise_pdf