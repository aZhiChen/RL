from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import numpy as np
import pandas as pd
from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = [None] * 4
        self.__target = [None] * 4
        self.__optimizer = [None] * 4
        for i in range(4):
            self.__policy[i] = DQN(action_dim, device).to(device) 
            self.__target[i] = DQN(action_dim, device).to(device)
            if restore is None:
                self.__policy[i].apply(DQN.init_weights)
            else:
                self.__policy[i].load_state_dict(torch.load(restore))
            self.__target[i].load_state_dict(self.__policy[i].state_dict())
            self.__optimizer[i] = optim.Adam(
                self.__policy[i].parameters(),
                lr=0.0000625,
                eps=1.5e-4,
            ) 
            self.__target[i].eval()
            
    def compute_q_next(self,batch):
        with torch.no_grad():
            q_min = self.__target[0](batch.next_state).clone()
            for i in range(1, 4):
                q = self.__target[i](batch.next_state)
                q_min = torch.min(q_min, q)
            q_next = q_min.max(1)[0]
            #q_target = batch.reward + self.discount * q_next * batch.mask
        return q_next

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        q_values = self.get_action_selection_q_values(state)
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                #return self.__policy[index](state).max(1).indices.item()
                return np.argmax(q_values)
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        # Choose a Q_net to udpate
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)
        #choose a Q randomly to update
        i = np.random.choice(4)
        values = self.__policy[i](state_batch.float()).gather(1, action_batch)
        
        #values_next = self.__target[0](next_batch.float()).max(1).values.detach()
        q_min = self.__target[0](reward_batch).clone().detach()
        for j in range(4): #k=4
            q = self.__target[i](reward_batch).detach()
            q_min = torch.min(q_min, q)
        value_next = q_min.max(1)[0]
        #find max min Q_i(s,a)
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer[i].zero_grad()
        loss.backward()
        for param in self.__policy[i].parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer[i].step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        for i in range(4):#k=4
            self.__target[i].load_state_dict(self.__policy[i].state_dict())

    def save(self, path: str) -> None:
        for i in range(4): #k=4
       # """save saves the state dict of the policy network."""
            torch.save(self.__policy[i].state_dict(), path)

            
    def get_action_selection_q_values(self, state):
        q_min = self.__policy[0](state)
        for i in range(4): #k=4
            q = self.__policy[i](state)
            q_min = torch.min(q_min, q)
        #q_min = to_numpy(q_min).flatten()
         #q_min = q_min.numpy().flatten()
        return q_min
            
