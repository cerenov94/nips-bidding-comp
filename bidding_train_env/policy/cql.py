import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math
import os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

class V(nn.Module):
    def __init__(self ,state ,hidden_dim,activation = 'tanh'):
        super().__init__()
        self.linear1 = layer_init(nn.Linear(state ,hidden_dim))
        self.activation = get_act_fn(activation)
        self.linear2 = layer_init(nn.Linear(hidden_dim,hidden_dim))
        self.linear3 = layer_init(nn.Linear(hidden_dim,1))

    def forward(self ,state):
        value = self.activation(self.linear1(state))
        value = self.activation(self.linear2(value))
        value = self.linear3(value)
        return value

class Q(nn.Module):
    def __init__(self ,state ,action ,hidden_dim,activation = 'relu'):
        super().__init__()
        self.activation = get_act_fn(activation)
        self.obs_emb = layer_init(nn.Linear(state,hidden_dim))
        self.act_emb = layer_init(nn.Linear(action,hidden_dim))
        self.linear1 = layer_init(nn.Linear(hidden_dim * 2,hidden_dim))
        self.linear2 = layer_init(nn.Linear(hidden_dim,hidden_dim))
        self.linear3 = layer_init(nn.Linear(hidden_dim,1))

    def forward(self ,state ,action):
        state = self.obs_emb(state)
        action = self.act_emb(action)
        sa = torch.cat([state,action],dim=-1)
        q = self.activation(self.linear1(sa))
        q = self.activation(self.linear2(q))
        q = self.linear3(q)
        return q




class Policy(nn.Module):
    def __init__(self, dim_obs=16, actions_dim=1, hidden_dim=64,activation = 'relu',n_layers = 2,dropout = 0.0):
        super().__init__()
        self.min_log_std = -10
        self.max_logs_std = 2
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = get_act_fn(activation)
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
        mu = self.mu(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std,self.min_log_std,self.max_logs_std)
        return mu,log_std

    def det_action(self,state):
        mu,std = self.forward(state)
        return mu,std

    def stoch_action(self,state):
        mu,log_std = self.forward(state)
        std = log_std.exp()
        pdf = Normal(mu,std)
        action = pdf.rsample()
        log_prob = pdf.log_prob(action)
        return action,log_prob



class CQL(nn.Module):

    def __init__(
            self,
            obs_shape,
            action_shape,
            tau,
            hidden_dim,
            lr,
            temp,
            with_lagrange,
            cql_weight,
            target_action_gap,
            device,
            batch_size
    ):
        super(CQL,self).__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.temp = temp
        self.with_lagrange = with_lagrange
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.target_entropy = -torch.FloatTensor([1.]).to(device)
        self.batch_size = batch_size
        # alpha
        self.log_alpha = torch.tensor([0.0],requires_grad=True,device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha],lr = self.lr)
        # cql
        self.cql_log_alpha = torch.zeros(1,requires_grad=True)
        self.cql_alpha_optimizer = torch.optim.Adam(params=[self.cql_log_alpha],lr = self.lr)
        # policy
        self.policy = Policy(self.obs_shape,self.action_shape,256,'relu',2,dropout=0.15)
        self.policy.to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),lr=self.lr)

        # critic
        # q1
        self.Q1 = Q(self.obs_shape,self.action_shape,self.hidden_dim)
        self.target_Q1 = Q(self.obs_shape,self.action_shape,self.hidden_dim)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.Q1.to(device)
        self.target_Q1.to(device)

        self.q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        # q2
        self.Q2 = Q(self.obs_shape,self.action_shape,self.hidden_dim)
        self.target_Q2 = Q(self.obs_shape,self.action_shape,self.hidden_dim)
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.Q2.to(device)
        self.target_Q2.to(device)

        self.q2_optimizer = torch.optim.Adam(self.Q2.parameters(),lr = self.lr)


    def get_action(self,state,deterministic = False):
        self.policy.eval()
        with torch.no_grad():
            if deterministic:
                action,_ = self.policy.det_action(state)
            else:
                action,_ = self.policy.stoch_action(state)

        return action

    def policy_loss(self,state,alpha):
        self.policy.train()
        action_pred,log_prob = self.policy.stoch_action(state)

        q1 = self.Q1(state,action_pred)
        q2 = self.Q2(state,action_pred)
        min_Q = torch.min(q1,q2)

        loss = ((alpha * log_prob - min_Q)).mean()

        return loss,log_prob

    def critic_loss(self,state_pi,state_q):
        action_pred,log_prob = self.policy.stoch_action(state_pi)
        q1 = self.Q1(state_q,action_pred)
        q2 = self.Q2(state_q,action_pred)
        loss1 = q1 - log_prob.detach()
        loss2 = q2 - log_prob.detach()

        return loss1,loss2

    def random_values(self,state,action,critic):
        random_qvalue = critic(state,action)
        log_prob = math.log(0.5 ** self.action_shape)

        return random_qvalue - log_prob

    def learn(self,replay_buffer):
        state, action, reward, next_state, next_action, not_done, G = replay_buffer.sample(self.batch_size)

        # policy train
        current_alpha = copy.deepcopy(self.alpha)
        policy_loss,log_prob = self.policy_loss(state,current_alpha)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # alpha loss
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # critic train
        self.policy.eval()
        with torch.no_grad():
            next_action_pred,next_log_p = self.policy.stoch_action(next_state)
            q1_next = self.target_Q1(next_state,next_action_pred)
            q2_next = self.target_Q2(next_state,next_action_pred)
            q_next = torch.min(q1_next,q2_next) - self.alpha.to(self.device) * next_log_p

            q_target = reward + (self.gamma * (1 - not_done) * q_next)

        q1 = self.Q1(state,action)
        q2 = self.Q2(state,action)

        q1_loss = F.mse_loss(q1,q_target)
        q2_loss = F.mse_loss(q2,q_target)

        # cql
        random_action = torch.FloatTensor(q1.shape[0] * 10,action.shape[-1]).uniform_(1,200).to(self.device)
        num_repeat = int(random_action.shape[0]/state.shape[0])

        temp_states = state.unsqueeze(1).repeat(1,num_repeat,1).view(state.shape[0] * num_repeat,state.shape[1])
        temp_next_states = next_state.unsqueeze(1).repeat(1,num_repeat,1).view(next_state.shape[0] * num_repeat,next_state.shape[1])

        current_pi1,current_pi2 = self.critic_loss(temp_states,temp_states)
        next_pi1,next_pi2 = self.critic_loss(temp_next_states,temp_next_states)

        random_values1 = self.random_values(temp_states,random_action,self.Q1).reshape(state.shape[0],num_repeat,1)
        random_values2 = self.random_values(temp_states,random_action,self.Q2).reshape(state.shape[0],num_repeat,1)

        current_pi1 = current_pi1.reshape(state.shape[0],num_repeat,1)
        current_pi2 = current_pi2.reshape(state.shape[0],num_repeat,1)

        next_pi1 = next_pi1.reshape(state.shape[0],num_repeat,1)
        next_pi2 = next_pi2.reshape(state.shape[0],num_repeat,1)

        cat_q1 = torch.cat([random_values1,current_pi1,next_pi1],1)
        cat_q2 = torch.cat([random_values2,current_pi2,next_pi2],1)

        assert cat_q1.shape == (state.shape[0],3 * num_repeat,1),f"cat q1 shape:{cat_q1.shape}"
        assert cat_q2.shape == (state.shape[0],3 * num_repeat,1),f"cat q2 shape:{cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1/self.temp,dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2/self.temp,dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight


        cql_alpha_loss = torch.FloatTensor([0.]).to(self.device)
        cql_alpha = torch.FloatTensor([0.]).to(self.device)
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(),0.0,1e5).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_q1_loss = q1_loss + cql1_scaled_loss
        total_q2_loss = q2_loss + cql2_scaled_loss

        # q1
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(),1)
        self.q1_optimizer.step()
        # q2
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(),1)
        self.q2_optimizer.step()

        self.soft_update(self.Q1,self.target_Q1)
        self.soft_update(self.Q2,self.target_Q2)


        return {
            'policy_loss':policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_scaled_loss.item(),
            'cql2_loss': cql2_scaled_loss.item(),
            'current_alpha': current_alpha.cpu().item(),
            'cql_alpha_loss': cql_alpha_loss.cpu().item(),
            'cql_alpha': cql_alpha.cpu().item()
        }

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def save_weights(self, save_path = "saved_model/CQLtest",name = 'cql_model'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        torch.save(self.policy.state_dict(), f'{save_path}/{name}.pth')
        torch.save(self.target_Q1.state_dict(),f'{save_path}/{name}_q1.pth')
        torch.save(self.target_Q2.state_dict(), f'{save_path}/{name}_q2.pth')
    def load_weights(self,load_path = 'saved_model/CQLtest'):
        self.policy.load_state_dict(torch.load(load_path,map_location='cpu'))

    def load_Q_weights(self, load_path='saved_model/CQLtest'):
        self.target_Q1.load_state_dict(torch.load(load_path + '/cql_model_q1.pth', map_location='cpu'))
        self.target_Q2.load_state_dict(torch.load(load_path + '/cql_model_q2.pth', map_location='cpu'))







