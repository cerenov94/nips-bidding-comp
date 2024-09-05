from copy import deepcopy
from torch.distributions import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

class Policy(nn.Module):
    def __init__(self, dim_obs=16, actions_dim=1, hidden_dim=64,max_action = 1):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            layer_init(nn.Linear(dim_obs, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim,hidden_dim)),
            nn.ReLU()
        )
        self.mu = layer_init(nn.Linear(hidden_dim,actions_dim))
        self.std = layer_init(nn.Linear(hidden_dim,actions_dim))

    def forward(self, state,deterministic = False,log_prob_out = False):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.std(x)
        log_std = torch.clip(log_std,-5,2)
        pdf = Normal(mu,torch.exp(log_std))
        if deterministic:
            action = mu
        else:
            action = pdf.rsample()

        log_prob = None
        #action = torch.clamp(action,min=0.)
        if log_prob_out:
            log_prob = pdf.log_prob(action).sum(-1)
        #print(action,log_prob)
        return action,log_prob

    def get_action(self,state):
        with torch.no_grad():
            action,_ = self.forward(state,deterministic=True)

        return action.cpu()



class EDAC:
    def __init__(
            self,
            obs_dim = 16,
            action_dim = 1,
            hidden_dim = 256,
            num_critics = 10,
            gamma = 0.99,
            tau = 5e-3,
            eta = 1.0,
            alpha_lr = 1e-4,
            batch_size = 32,
            device = 'cpu'
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.eta = eta
        self.num_critics = num_critics
        self.batch_size = batch_size
        # policy
        self.policy = Policy(dim_obs=self.obs_dim,actions_dim=self.action_dim,hidden_dim=self.hidden_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        # critic
        self.critic = VectorizedCritic(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_critics=self.num_critics
        ).to(self.device)
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic).to(device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = 1e-4)
        # alpha
        self.alpha_lr = alpha_lr
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.tensor([0.0],dtype=torch.float32,device=self.device,requires_grad=True)
        self.alph_optimizer = torch.optim.Adam([self.log_alpha],lr = self.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

    def alpha_loss(self,state):
        with torch.no_grad():
            action,action_log_prob = self.policy(state,log_prob_out = True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()
        return loss

    def actor_loss(self,state):
        action,action_log_prob = self.policy(state,log_prob_out = True)
        q_values = self.critic(state,action)
        q_value_min = q_values.min(0).values
        q_value_std = q_values.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss,batch_entropy,q_value_std

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


        q_values = self.critic(state,action)

        q_action_grad = torch.autograd.grad(
            q_values.sum(),action,retain_graph=True,create_graph=True
        )[0]
        q_action_grad = q_action_grad/(torch.norm(q_action_grad,p=2,dim=2).unsqueeze(-1) + 1e-10)
        q_action_grad = q_action_grad.transpose(0, 1)
        masks = (torch.eye(num_critics,device=self.device).unsqueeze(dim=0).repeat(q_action_grad.shape[0],1,1))

        q_action_grad = q_action_grad @ q_action_grad.permute(0,2,1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1,2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss

    def critic_loss(
            self,
            state,
            action,
            reward,
            next_state,
            done
    ):
        with torch.no_grad():
            next_action,next_action_log_prob = self.policy(next_state,log_prob_out = True)
            q_next = self.target_critic(next_state,next_action).min(0).values
            #print(q_next.shape,next_action_log_prob.sum().shape)
            q_next = q_next + self.alpha * next_action_log_prob
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(dim= - 1)

        q_values = self.critic(state,action)
        critic_loss = ((q_values - q_target.view(1,-1)) ** 2).mean(dim= 1).sum(dim=0)
        diversity_loss = self.critic_diversity_loss(state,action)
        loss = critic_loss + self.eta * diversity_loss

        return loss

    def train(self,replay_buffer):
        state, action, reward, next_state, next_action, done, G = replay_buffer.sample(self.batch_size)

        # alpha learn
        alpha_loss = self.alpha_loss(state)
        self.alph_optimizer.zero_grad()
        alpha_loss.backward()
        self.alph_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # actor learn
        actor_loss,actor_batch_entropy,q_policy_std = self.actor_loss(state)
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # critic learn
        critic_loss = self.critic_loss(state,action,reward,next_state,done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            max_action = self.policy.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)
            q_random_std = self.critic(state,random_actions).std(0).mean(0).item()

        return (
            alpha_loss.item(),
            critic_loss.item(),
            actor_loss.item(),
            actor_batch_entropy,
            self.alpha.item(),
            q_policy_std,
            q_random_std
        )
    def get_action(self,state):
        action = self.policy.get_action(state)
        action = torch.clamp(action,min=0.)
        return action

    def save_weights(self, save_path = "saved_model/EDACtest"):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        torch.save(self.policy.to('cpu').state_dict(), f'{save_path}/edac_model.pth')

    def load_weights(self,load_path = 'saved_model/EDACtest'):
        self.policy.load_state_dict(torch.load(load_path,map_location='cpu'))