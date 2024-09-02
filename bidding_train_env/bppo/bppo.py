import torch
import torch.nn as nn
from torch.distributions import  Normal
import torch.nn.functional as F
from copy import deepcopy
import os

class V(nn.Module):
    def __init__(self ,state ,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(state ,hidden_dim * 2)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim * 2,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim ,hidden_dim)
        self.linear4 = nn.Linear(hidden_dim,1)

    def forward(self ,state):
        value = self.linear4(self.activation(self.linear3(self.activation(self.linear2(self.activation(self.linear1(state)))))))
        return value

class Q(nn.Module):
    def __init__(self ,state ,action ,hidden_dim):
        super().__init__()

        self.obs_emb = nn.Linear(state,hidden_dim)
        self.act_emb = nn.Linear(action,hidden_dim)
        self.linear1 = nn.Linear(hidden_dim * 2,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,1)

    def forward(self ,state ,action):
        state = self.obs_emb(state)
        action = self.act_emb(action)
        sa = torch.cat([state,action],dim=-1)
        qv = self.linear3(F.relu(self.linear2(F.relu(self.linear1(sa)))))
        return qv


class Policy(nn.Module):
    def __init__(self, dim_obs=16, actions_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actions_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, 1))

    def forward(self, state):
        mu = self.net(state)
        log_std = self.actor_log_std.expand_as(mu)
        actor_std = torch.exp(log_std)
        pdf = Normal(mu, actor_std)
        return pdf

    def det_action(self,state):
        mu = self.net(state)
        return mu


class Value:
    def __init__(self, dim_obs=16, hidden_dim=64, lr=1e-4, batch_size=4,device = 'cpu'):
        super().__init__()

        self.value_net = V(dim_obs, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.batch_size = batch_size

    def __call__(self, state):
        return self.value_net(state)

    def train(self, replay_buffer,Q1,Q2):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            q1 = Q1(state,action)
            q2 = Q2(state,action)
            min_Q = torch.min(q1,q2)
        value = self.value_net(state)

        loss = self.l2_loss(min_Q-value,0.7).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)


class QL:
    def __init__(
            self,
            dim_obs=16,
            dim_actions=1,
            hidden_dim=128,
            lr=1e-4,
            update_freq=200,
            tau=5e-3,
            gamma=0.99,
            batch_size=4,
            device = 'cpu'
    ):
        super().__init__()
        self.Q = Q(dim_obs, dim_actions, hidden_dim).to(device)
        self.target_Q = Q(dim_obs, dim_actions, hidden_dim).to(device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0

    def __call__(self, state, action):
        return self.Q(state, action)

    def loss(self, replay_buffer, p=None):
        raise NotImplementedError

    def train(self, replay_buffer, p=None):
        q_loss = self.loss(replay_buffer, p)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            # print('updating')
            for param,target_param in zip(self.Q.parameters(),self.target_Q.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            #self.target_Q.load_state_dict(self.Q.state_dict())

        return q_loss.item()


class QLSarsa(QL):
    def __init__(
            self,
            dim_obs=16,
            dim_actions=1,
            hidden_dim=128,
            lr=1e-4,
            update_freq=200,
            tau=5e-3,
            gamma=0.99,
            batch_size=4,
            device = 'cpu'
    ):
        super().__init__()
        self.Q = Q(dim_obs, dim_actions, hidden_dim).to(device)
        self.target_Q = Q(dim_obs, dim_actions, hidden_dim).to(device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0

    def loss(self, replay_buffer, p=None):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            q_target_value = reward + (1 - done) * self.gamma * self.target_Q(next_state, next_action)

        q_value = self.Q(state, action)
        loss = F.mse_loss(q_value, q_target_value)

        return loss


class QP(QL):
    def __init__(self, dim_obs=16, dim_actions=1, hidden_dim=128, lr=1e-4, update_freq=200, tau=5e-3, gamma=0.99,
                 batch_size=4,device = 'cpu'):
        super().__init__(
            tau=tau,
            update_freq=update_freq,
            gamma=gamma,
            batch_size=batch_size,
            hidden_dim = hidden_dim,
            lr = lr,
            device=device
        )

    def loss(self, replay_buffer, p=None):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)
        next_a = p.get_action(next_state)
        with torch.no_grad():
            q_target_value = reward + (1 - done) * self.gamma * self.target_Q(next_state, next_a.to(state.device))
        q_value = self.Q(state, action)
        loss = F.mse_loss(q_value, q_target_value)

        return loss


class BC:
    def __init__(self, dim_obs=16, dim_actions=1, hidden_dim=64, lr=1e-4, batch_size=4,device = 'cpu'):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim).to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def loss(self, replay_buffer):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)

        pdf = self.policy(state)
        log_prob = pdf.log_prob(action)
        loss = (-log_prob).mean()
        return loss

    def train(self, replay_buffer):
        loss = self.loss(replay_buffer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state):
        pdf = self.policy(state)
        action = pdf.rsample()
        return action.detach().cpu()


class PPO:
    def __init__(self, dim_obs=16, dim_actions=1, hidden_dim=64, lr=1e-4, clip_ratio=0.25, entropy=0., decay=0.96,
                 omega=0.9, batch_size=4,device = 'cpu'):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.old_policy = deepcopy(self.policy).to(device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.clip_ratio = clip_ratio
        self.decay = decay
        self.omega = omega
        self.batch_size = batch_size
        self.entropy = entropy

    def loss(
            self,
            replay_buffer,
            Q1=None,
            Q2=None,
            V=None,
            is_clip_decay=None,
            is_lr_decay=None
    ):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)
        old_dist = self.old_policy(state)
        new_dist = self.policy(state)
        new_log_prob = new_dist.log_prob(action)
        old_log_rpob = old_dist.log_prob(action)
        ratio = (new_log_prob - old_log_rpob).exp()
        loss1 = ratio * G
        if is_clip_decay:
            self.clip_ratio = self.clip_ratio * self.decay
        else:
            self.clip_ratio = self.clip_ratio

        loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * G

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self.entropy

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss

    def train(self, replay_buffer, Q1=None,Q2=None, V=None, is_clip_decay=None, is_lr_decay=None):
        loss = self.loss(replay_buffer, Q1,Q2, V, is_clip_decay, is_lr_decay)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        if is_lr_decay:
            self.scheduler.step()
        return loss.item()

    def get_action(self, state):
        with torch.no_grad():
            action = self.policy.det_action(state)
        action = action.detach().clamp(min=0).cpu()
        return action


class BPPO(PPO):
    def __init__(self, dim_obs=16, dim_actions=1, hidden_dim=64, lr=1e-4, clip_ratio=0.25, entropy=0., decay=0.96,
                 omega=0.9, batch_size=32,device = 'cpu'):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.old_policy = deepcopy(self.policy).to(device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.clip_ratio = clip_ratio
        self.decay = decay
        self.omega = omega
        self.batch_size = batch_size
        self.entropy = entropy

    def loss(self, replay_buffer, Q1,Q2, V, is_clip_decay, is_lr_decay):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)

        old_dist = self.old_policy(state)
        a = old_dist.rsample()
        min_Q = torch.min(Q1(state,a),Q2(state,a))
        advantage = min_Q - V(state)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        new_dist = self.policy(state)

        new_log_prob = new_dist.log_prob(a)
        old_log_prob = old_dist.log_prob(a)
        ratio = (new_log_prob - old_log_prob).exp()

        loss1 = ratio * advantage

        if is_clip_decay:
            self.clip_ratio = self.clip_ratio * self.decay
        else:
            self.clip_ratio = self.clip_ratio

        loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self.entropy

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss

    def save_weights(self, save_path = "saved_model/BPPOtest"):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        torch.save(self.policy.to('cpu').state_dict(), f'{save_path}/bppo_model.pth')

    def load_weights(self,load_path = 'saved_model/BPPOtest'):
        self.policy.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.old_policy.load_state_dict(torch.load(load_path,map_location='cpu'))