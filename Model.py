import torch.nn as nn
import torch

class NetModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(NetModel, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.fc0 = nn.Linear(state_dim, 50)
        self.fc1 = nn.Linear(50, 64)
        self.fc = nn.Linear(64, act_dim)

    def forward(self, obs):
        out = self.fc0(obs)
        out = torch.tanh(out)
        out = self.fc1(out)
        out = torch.tanh(out)
        out = self.fc(out)
        return out

class DuelingDqn(nn.Module):
    def __init__(self, state_dim, act_dim,embed_size=1):
        super(DuelingDqn, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.fc0 = nn.Linear(state_dim, 256)
        self.fc1 = nn.Linear(256, 128)
        # origin DDQN
        # self.fc = nn.Linear(64, act_dim)
        # Dueling DQN
        # 状态值函数V
        self.valueFc = nn.Linear(128, 1)
        # 优势函数A
        self.advantageFc = nn.Linear(128, act_dim)

    def forward(self, obs):
        out = self.fc0(obs)
        out = torch.relu(out)
        out = self.fc1(out)
        out = torch.relu(out)
        V = self.valueFc(out)
        advantage = self.advantageFc(out)
        # 计算优势函数的均值,用于归一化
        advMean = torch.mean(advantage, dim=1, keepdim=True)
        # 状态行为值函数Q=V+A
        Q = advantage + (V - advMean)
        return Q
    

class DuelingDqn_embed(nn.Module):
    def __init__(self, state_dim, act_dim,embed_size=4):
        super(DuelingDqn_embed, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.embed_size = embed_size
        self.embed = nn.Embedding(9,embed_size)


        self.fc0 = nn.Linear(state_dim*embed_size, 256)
        self.fc1 = nn.Linear(256, 128)
        # origin DDQN
        # self.fc = nn.Linear(64, act_dim)
        # Dueling DQN
        # 状态值函数V
        self.valueFc = nn.Linear(128, 1)
        # 优势函数A
        self.advantageFc = nn.Linear(128, act_dim)

    def forward(self, obs):
        obs = obs.long()
        out = self.embed(obs)
        out = torch.reshape(out,(-1,self.state_dim*self.embed_size))
        out = self.fc0(out)
        out = torch.relu(out)
        out = self.fc1(out)
        out = torch.relu(out)
        V = self.valueFc(out)
        advantage = self.advantageFc(out)
        # 计算优势函数的均值,用于归一化
        advMean = torch.mean(advantage, dim=1, keepdim=True)
        # 状态行为值函数Q=V+A
        Q = advantage + (V - advMean)
        return Q