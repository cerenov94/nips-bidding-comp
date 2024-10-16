import random
from collections import namedtuple
import numpy as np
import torch
from operator import itemgetter


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state","next_action","done","G"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self,device = 'cpu'):
        self.memory = []
        self.device = device


    def push(self, state, action, reward, next_state,next_action,done,G):
        """saving an experience tuple"""
        experience = Experience(state, action, reward, next_state, next_action,done,G)
        self.memory.append(experience)

    def sample(self, batch_size,random_samples = True):
        """randomly sampling a batch of experiences"""
        if random_samples:
            tem = random.sample(self.memory, batch_size)
        else:
            tem = self.memory[:]
        states, actions, rewards, next_states,next_action,dones,G = zip(*tem)
        states, actions, rewards, next_states,next_action,dones,G = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(
            next_states),np.stack(next_action),np.stack(dones),np.stack(G)
        states, actions, rewards, next_states,next_action,dones,G = torch.FloatTensor(states), torch.FloatTensor(
            actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states),torch.FloatTensor(next_action),torch.FloatTensor(dones),torch.FloatTensor(G)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_action = next_action.to(self.device)
        dones = dones.to(self.device)
        G = G.to(self.device)

        return states, actions, rewards, next_states,next_action,dones,G

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.memory)

    # def split_memory(self,flag = True):
    #     if flag:
    #         self.valid_memory = self.memory[12576:12624]
    #         self.memory = self.memory[:12576] + self.memory[12624:]
    #     else:
    #         self.valid_memory = self.memory[12576:12624]


class NstepRB:
    def __init__(self, df, n_steps=3,device = 'cpy'):

        self.n_steps = n_steps
        state = np.stack(df['state'].values)
        action = np.expand_dims(np.stack(df['action'].values), axis=-1)
        reward = np.expand_dims(np.stack(df['reward'].values), axis=-1)
        done = np.expand_dims(np.stack(df['done'].values), axis=-1)

        self.episodes = {}

        states, actions, rewards, dones = [], [], [], []
        self.episode_counter = 0
        for i in range(len(done)):
            states.append(state[i])
            actions.append(action[i])
            rewards.append(reward[i])
            dones.append(done[i])
            if done[i]:
                if len(states) != 1:
                    self.episodes[self.episode_counter] = {
                        'state': torch.from_numpy(np.stack(states)).to(torch.float32).to(device),
                        'action': torch.from_numpy(np.stack(actions)).to(torch.float32).to(device),
                        'reward': torch.from_numpy(np.stack(rewards)).to(torch.float32).to(device),
                        'done': torch.from_numpy(np.stack(dones))
                    }
                    self.episode_counter += 1
                states, actions, rewards, dones = [], [], [], []
        self.memory = {}
