import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """saving an experience tuple"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """randomly sampling a batch of experiences"""
        tem = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*tem)
        states, actions, rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(
            next_states), np.stack(dones)
        states, actions, rewards, next_states, dones = torch.FloatTensor(states), torch.FloatTensor(
            actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.memory)


class PrioritizedReplayBuffer():
    def __init__(self, max_capacity=100000, batch_size=4):
        self.max_capacity = max_capacity
        self.memory = []
        self.batch_size = batch_size
        self.priority = np.zeros((self.max_capacity), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        max_priority = self.priority.max() if self.memory else 1.0
        if len(self.memory) < self.max_capacity:
            self.memory.append([state, action, reward, next_state, done])
        else:
            self.memory[self.pos] = [state, action, reward, next_state, done]

        self.priority[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.max_capacity

    def clear_histoy(self):
        self.memory.clear()

    def sample(self,BATCH_SIZE,alpha = 0.5, beta = 0.5):
        if len(self.memory) == self.max_capacity:
            p = self.priority
        else:
            p = self.priority[:self.pos]

        probas = p ** alpha
        probas = probas / probas.sum()
        sample_idxs = np.random.choice(len(self.memory), BATCH_SIZE, p=probas)
        sample = [self.memory[x] for x in sample_idxs]

        weights = (len(self.memory) * probas[sample_idxs]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        state, action, reward, next_state, done = zip(*sample)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, weights, sample_idxs

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    buffer = PrioritizedReplayBuffer()
    for i in range(1000):
        buffer.push(np.array([1, 2, 3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    print(buffer.sample(20))
