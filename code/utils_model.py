import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1_value = nn.Linear(64*7*7, 512)
        self.__fc1_adv = nn.Linear(64*7*7, 512)
        self.__fc2_value = nn.Linear(512, 1)
        self.__fc2_adv = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        j = x / 255.
        j = F.relu(self.__conv1(j))
        j = F.relu(self.__conv2(j))
        j = F.relu(self.__conv3(j))
        value = F.relu(self.__fc1_value(j.view(x.size(0), -1)))
        adv = F.relu(self.__fc1_adv(j.view(x.size(0), -1)))
        value = self.__fc2_value(value)
        adv = self.__fc2_adv(adv)
        Q = value.expand_as(adv) + adv
        advAverage = torch.mean(adv,dim=1,keepdim=True)
        Q = (Q - advAverage.expand_as(adv))
        #x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return Q

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
