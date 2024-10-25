import torch
import torch.nn as nn
from torch.distributions import  Normal
import torch.nn.functional as F
from copy import deepcopy
import os
import numpy as np
import math



class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.obs_emb = VectorizedLinear(state_dim,hidden_dim,num_critics)
        self.act_emb = VectorizedLinear(action_dim,hidden_dim,num_critics)
        self.critic = nn.Sequential(
            layer_init(VectorizedLinear(hidden_dim * 2, hidden_dim, num_critics)),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            layer_init(VectorizedLinear(hidden_dim, 1, num_critics)),
        )
        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        #state_action = torch.cat([state, action], dim=-1)
        #state_action = None
        if state.dim() != 3 and action.dim() != 3:
            assert state.dim() == 2 and action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state = state.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
            action = action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        state = self.obs_emb(state)
        action = self.act_emb(action)
        state_action = torch.cat([state,action],dim=-1)
        #print(state_action.shape)
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class V(nn.Module):
    def __init__(self ,state ,hidden_dim,activation = 'tanh'):
        super().__init__()
        self.linear1 = layer_init(nn.Linear(state ,hidden_dim * 2))
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        self.linear2 = layer_init(nn.Linear(hidden_dim * 2,hidden_dim))
        self.linear3 = layer_init(nn.Linear(hidden_dim ,hidden_dim))
        self.linear4 = layer_init(nn.Linear(hidden_dim,1))

    def forward(self ,state):
        value = self.linear4(self.activation(self.linear3(self.activation(self.linear2(self.activation(self.linear1(state)))))))
        return value

class Q(nn.Module):
    def __init__(self ,state ,action ,hidden_dim,activation = 'relu'):
        super().__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        self.obs_emb = layer_init(nn.Linear(state,hidden_dim))
        self.act_emb = layer_init(nn.Linear(action,hidden_dim))
        self.linear1 = layer_init(nn.Linear(hidden_dim * 2,hidden_dim))
        self.linear2 = layer_init(nn.Linear(hidden_dim,hidden_dim))
        self.linear3 = layer_init(nn.Linear(hidden_dim,1))

    def forward(self ,state ,action):
        state = self.obs_emb(state)
        action = self.act_emb(action)
        sa = torch.cat([state,action],dim=-1)
        qv = self.linear3(self.activation(self.linear2(self.activation(self.linear1(sa)))))
        return qv


def get_act_fn(activation):
    if activation == 'swish':
        return nn.SiLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()

class Policy(nn.Module):
    def __init__(self, dim_obs=16, actions_dim=1, hidden_dim=64,activation = 'relu',n_layers = 2,dropout = 0.0):
        super().__init__()
        self.min_log_std = -10
        self.max_logs_std = 2
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = get_act_fn(activation)

        # self.net = nn.Sequential(
        #     layer_init(nn.Linear(dim_obs, hidden_dim)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(hidden_dim, hidden_dim)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(hidden_dim,hidden_dim)),
        #     nn.Tanh(),
        # )
        self.emb = nn.Linear(dim_obs,hidden_dim)
        layers  = []
        for i in range(self.n_layers):
            layers.append(layer_init(nn.Linear(hidden_dim,hidden_dim)))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout))
        self.net = nn.Sequential(*layers)
        self.mu = layer_init(nn.Linear(hidden_dim,actions_dim))
        self.std = layer_init(nn.Linear(hidden_dim,actions_dim))

    def forward(self, state):
        x = self.activation(self.emb(state))
        x = self.net(x)
        #x = self.net(state)
        mu = self.mu(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std,self.min_log_std,self.max_logs_std)
        return mu,log_std

    def det_action(self,state):
        mu,std = self.forward(state)
        return mu,std

    def get_pdf(self,state):
        mu,log_std = self.forward(state)
        std = log_std.exp()
        pdf = Normal(mu,std)
        return pdf


class Value:
    def __init__(self, dim_obs=16, hidden_dim=64, lr=1e-4, batch_size=4,device = 'cpu',expectile = 0.7):
        super().__init__()
        self.device = device
        self.value_net = V(dim_obs, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.expectile = expectile

    def __call__(self, state):
        return self.value_net(state)

    def train(self, replay_buffer,Q1,Q2):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            q1 = Q1(state, action)
            q2 = Q2(state, action)
            min_Q = torch.min(q1, q2)
        value = self.value_net(state)

        loss = self.l2_loss(min_Q - value, self.expectile).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * diff.pow(2)

    def save_weights(self, save_path = "saved_model/BPPOtest",):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.value_net.to('cpu').state_dict(), f'{save_path}/value_model.pth')

    def load_weights(self,load_path = 'saved_model/BPPOtest'):
        self.value_net.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.value_net.to(self.device)


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
        self.device = device
        self.Q = Q(dim_obs, dim_actions, hidden_dim).to(self.device)
        self.target_Q = Q(dim_obs, dim_actions, hidden_dim).to(self.device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0

    def __call__(self, state, action):
        return self.Q(state, action)

    def loss(self, replay_buffer, p=None,V = None):
        raise NotImplementedError

    def train(self, replay_buffer, p=None,V = None):
        q_loss = self.loss(replay_buffer, p,V)
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

    def save_weights(self, save_path = "saved_model/BPPOtest",name = "Q1"):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.Q.to('cpu').state_dict(), f'{save_path}/{name}.pth')

    def load_weights(self,load_path = 'saved_model/BPPOtest'):
        print('LOAD Q WEIGHTS')
        self.Q.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.target_Q.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.Q.to(self.device)
        self.target_Q.to(self.device)


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
            device = 'cpu',

    ):
        super().__init__(
            dim_obs=dim_obs,
            dim_actions=dim_actions,
            hidden_dim=hidden_dim,
            lr=lr,
            update_freq=update_freq,
            tau=tau,
            gamma=gamma,
            batch_size=batch_size,
            device=device
        )
        self.Q = Q(dim_obs, dim_actions, hidden_dim).to(self.device)
        self.target_Q = Q(dim_obs, dim_actions, hidden_dim).to(self.device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.AdamW(self.Q.parameters(), lr=lr,weight_decay=1e-3)
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0

    def loss(self, replay_buffer, p=None,V = None):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            #q_target_value = reward + (1 - done) * self.gamma * self.target_Q(next_state, next_action)
            q_target_value = reward + (1 - done) * self.gamma * V(next_state)
        q_value = self.Q(state, action)
        loss = F.mse_loss(q_value, q_target_value)

        return loss


class QP(QL):
    def __init__(self, dim_obs=16, dim_actions=1, hidden_dim=128, lr=1e-4, update_freq=200, tau=5e-3, gamma=0.99,
                 batch_size=4,device = 'cpu'):
        super().__init__(
            dim_obs = dim_obs,
            dim_actions=dim_actions,
            tau=tau,
            update_freq=update_freq,
            gamma=gamma,
            batch_size=batch_size,
            hidden_dim = hidden_dim,
            lr = lr,
            device=device
        )

    def loss(self, replay_buffer, p=None,V = None):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)
        pdf = p.policy.get_pdf(next_state)
        next_a = pdf.rsample()
        with torch.no_grad():
            q_target_value = reward + (1 - done) * self.gamma * self.target_Q(next_state,next_a.to(next_state.device))
        q_value = self.Q(state, action)
        loss = F.mse_loss(q_value, q_target_value)

        return loss


class QLVect(QL):
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
            num_critics = 10,
            device = 'cpu',
            eta = 1.0
    ):
        super().__init__()
        self.Q = VectorizedCritic(state_dim=dim_obs,action_dim=dim_actions,hidden_dim=hidden_dim,num_critics=num_critics).to(device)
        with torch.no_grad():
            self.target_Q = deepcopy(self.Q).to(device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0
        self.eta = eta
        self.num_critics = num_critics
        self.device = device

    def loss(self, replay_buffer, p=None,V = None):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            #q_target_value = reward + (1 - done) * self.gamma * self.target_Q(next_state, next_action).min(0).values.unsqueeze(dim= - 1)
            q_target_value = reward + (1 - done) * self.gamma * V(next_state)
        q_value = self.Q(state, action)
        loss = (q_value - q_target_value.view(1,-1)).pow(2).mean(dim = 1).sum(dim = 0)
        diversity_loss = self.critic_diversity_loss(state, action)
        loss = loss + self.eta * diversity_loss
        return loss

    def critic_diversity_loss(
            self,
            state,
            action
    ):
        num_critics = self.num_critics
        state = state.unsqueeze(dim=0).repeat_interleave(num_critics,dim=0)
        action = (
            action.unsqueeze(dim=0).repeat_interleave(num_critics,dim=0).requires_grad_(True)
        )
        q_values = self.Q(state,action)


        q_action_grad = torch.autograd.grad(
            q_values.sum(),action,retain_graph=True,create_graph=True,
        )[0]

        q_action_grad = q_action_grad/(torch.norm(q_action_grad,p=2,dim=2).unsqueeze(-1) + 1e-10)
        q_action_grad = q_action_grad.transpose(0, 1)
        masks = (torch.eye(num_critics,device=self.device).unsqueeze(dim=0).repeat(q_action_grad.shape[0],1,1))

        q_action_grad = q_action_grad @ q_action_grad.permute(0,2,1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1,2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss


class BC:
    def __init__(
            self,
            dim_obs=16,
            dim_actions=1,
            hidden_dim=64,
            lr=1e-4,
            batch_size=4,
            device = 'cpu',
            temp = 3.0,
            activation = 'relu',
            n_layers = 2,
            dropout = 0.):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim,activation,n_layers,dropout).to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp = temp
    def loss(self, replay_buffer,Q1,Q2,V):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            v = V(state)
            q1 = Q1(state, action)
            q2 = Q2(state, action)
            min_Q = torch.min(q1, q2)
        exp_a = torch.exp(min_Q- v) * self.temp
        exp_a = torch.min(exp_a,torch.FloatTensor([100.0]).to(exp_a.device))

        pdf = self.policy.get_pdf(state)
        log_prob = pdf.log_prob(action)
        loss =  -(log_prob * exp_a).mean()
        return loss

    def train(self, replay_buffer,Q1,Q2,V):
        self.policy.train()
        loss = self.loss(replay_buffer,Q1,Q2,V)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            pdf = self.policy.get_pdf(state)
            action = pdf.rsample()
        return action.detach().cpu()


class PPO:
    def __init__(
            self,
            dim_obs=16,
            dim_actions=1,
            hidden_dim=64,
            lr=1e-4,
            clip_ratio=0.25,
            entropy=0.,
            decay=0.96,
            omega=0.9,
            batch_size=4,
            device = 'cpu',
            activation = 'relu',
            n_layers = 2,
            dropout = 0.
    ):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim,activation,n_layers,dropout).to(device)
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
            Q2 = None,
            V=None,
            is_clip_decay=None,
            is_lr_decay=None
    ):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)
        self.old_policy.eval()
        with torch.no_grad():
            old_dist = self.old_policy.get_pdf(state)
        new_dist = self.policy.get_pdf(state)
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
        self.policy.train()
        loss,approx_kl = self.loss(replay_buffer, Q1,Q2, V, is_clip_decay, is_lr_decay)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        if is_lr_decay:
            self.scheduler.step()
        return loss.item(),approx_kl

    def get_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            action,std = self.policy.det_action(state)
        action = torch.clamp(action, 0)
        action = action.detach().cpu()
        return action,std.detach().cpu()

    def weighted_advantage(self,advantage):

        if self.omega == 0.5:
            return advantage
        else:
            weight = torch.zeros_like(advantage)
            index = torch.where(advantage > 0)[0]
            weight[index] = self.omega
            weight[torch.where(weight == 0)[0]] = 1 - self.omega
            weight.to(advantage.device)
            return weight * advantage

class BPPO(PPO):
    def __init__(
            self,
            dim_obs=16,
            dim_actions=1,
            hidden_dim=64,
            lr=1e-4,
            clip_ratio=0.25,
            entropy=0.,
            decay=0.96,
            omega=0.9,
            batch_size=32,
            device = 'cpu',
            n_steps = 14000,
            activation = 'relu',
            n_layers = 2,
            dropout = 0.
    ):
        super().__init__()
        self.policy = Policy(dim_obs, dim_actions, hidden_dim,activation,n_layers,dropout).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.old_policy = deepcopy(self.policy).to(device)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=n_steps,eta_min=0.0)
        self.clip_ratio = clip_ratio
        self.decay = decay
        self.omega = omega
        self.batch_size = batch_size
        self.entropy = entropy

    def loss(
            self,
            replay_buffer,
            Q1 = None,
            Q2 = None,
            V = None,
            is_clip_decay = None,
            is_lr_decay = None
    ):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)

        self.old_policy.eval()
        with torch.no_grad():
            v = V(state)
            q1 = Q1(state, action)
            q2 = Q2(state, action)
            min_Q = torch.min(q1, q2)
            advantage = min_Q - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            old_dist = self.old_policy.get_pdf(state)
            a = old_dist.rsample()

        self.weighted_advantage(advantage)

        new_dist = self.policy.get_pdf(state)

        new_log_prob = new_dist.log_prob(a)
        old_log_prob = old_dist.log_prob(a)

        log_ratio = new_log_prob - old_log_prob
        ratio = log_ratio.exp()

        with torch.no_grad():
            old_approx_kl = (-log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()


        loss1 = ratio * advantage

        if is_clip_decay:
            self.clip_ratio = self.clip_ratio * self.decay
        else:
            self.clip_ratio = self.clip_ratio

        loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self.entropy

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss,approx_kl

    def save_weights(self, save_path = "saved_model/BPPOtest",old = False):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if old:
            torch.save(self.old_policy.to('cpu').state_dict(), f'{save_path}/bppo_model.pth')
        else:
            torch.save(self.policy.to('cpu').state_dict(), f'{save_path}/bppo_model.pth')

    def load_weights(self,load_path = 'saved_model/BPPOtest'):
        self.policy.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.old_policy.load_state_dict(torch.load(load_path,map_location='cpu'))