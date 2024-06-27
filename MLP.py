import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):    
    def __init__(self, in_features, out_features, act_fn):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = act_fn()
        
    def forward(self, x):              
        y = self.act_fn(self.fc(x))
        return y

    
class OutputBlock(nn.Module):
    def __init__(self, latent_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.fc_r = nn.Linear(latent_dim, (self.num_agents + 1) * self.num_agents)
        self.fc_c = nn.Linear(latent_dim, self.num_agents * (self.num_agents + 1))

    def forward(self, x, mask_p, mask_q):
        
        row = self.fc_r(x).view(-1, self.num_agents + 1, self.num_agents)
        col = self.fc_c(x).view(-1, self.num_agents, self.num_agents + 1)
        
        row = F.softplus(row) * mask_p
        col = F.softplus(col) * mask_q
        
        row = F.normalize(row, p = 1, dim = 1, eps=1e-8)[:, :-1, :]
        col = F.normalize(col, p = 1, dim = 2, eps=1e-8)[:, :, :-1]

        return torch.minimum(row, col)
    
    
class Net(nn.Module):
    def __init__(self, net_arch, act_fn, num_agents):
        super().__init__()
        
        blocks = []
        self.num_agents = num_agents
        last_layer_dim = 2 * (self.num_agents**2)
        
        for curr_layer_dim in net_arch:
            blocks.append(DenseBlock(last_layer_dim, curr_layer_dim, act_fn))
            last_layer_dim = curr_layer_dim
              
        self.network = nn.Sequential(*blocks)
        self.output = OutputBlock(last_layer_dim, self.num_agents)             

        
    def forward(self, p, q):
        p = torch.relu(p)
        q = torch.relu(q)
        mask_p = torch.nn.functional.pad((p > 0).to(p.dtype), (0, 0, 0, 1, 0, 0), mode='constant', value=1)
        mask_q = torch.nn.functional.pad((q > 0).to(q.dtype), (0, 1, 0, 0, 0, 0), mode='constant', value=1)
        
        x = torch.stack([p, q], dim = 1).view(-1, 2 * (self.num_agents**2))
        x = self.network(x)
        return self.output(x, mask_p, mask_q)