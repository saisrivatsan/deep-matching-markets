import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1)
        self.conv_max_p = nn.Conv2d(c_in, c_out, 1)
        self.conv_max_q = nn.Conv2d(c_in, c_out, 1)
        
    def forward(self, x):
        y = self.conv(x)
        y = y + self.conv_max_p(x.max(dim = -1, keepdim = True)[0])
        y = y + self.conv_max_q(x.max(dim = -2, keepdim = True)[0])
        return y

    
class ConvBlock(nn.Module):    
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.conv = ConvLayer(c_in, c_out)
        self.act_fn = act_fn()
        self.norm_fn = nn.GroupNorm(c_in, c_in)
        self.res_cxn = (c_in == c_out)
       
    def forward(self, x):        
        x = self.norm_fn(x)        
        y = self.act_fn(self.conv(x))
        if self.res_cxn: y = y + x
        return y

    
class OutputBlock(nn.Module):
    def __init__(self, c_in, act_fn):
        super().__init__()
        self.layer = ConvBlock(c_in, c_out = 4, act_fn = act_fn)

    def forward(self, x, mask_p, mask_q):

        y = self.layer(x)
        
        y1, y2, ydp, ydq = torch.split(y, 1, dim = 1)
        
        ydp = ydp.mean(dim = -2, keepdim = True)
        row = torch.cat([y1, ydp], dim = -2).squeeze(1)
        
        ydq = ydq.mean(dim = -1, keepdim = True)
        col = torch.cat([y2, ydq], dim = -1).squeeze(1)

        row = F.softplus(row) * mask_p
        col = F.softplus(col) * mask_q
        
        row = F.normalize(row, p = 1, dim = 1, eps=1e-8)[:, :-1, :]
        col = F.normalize(col, p = 1, dim = 2, eps=1e-8)[:, :, :-1]

        return torch.minimum(row, col)
    
    
class Net(nn.Module):
    def __init__(self, net_arch, act_fn):
        super().__init__()
        
        blocks = []
        last_layer_dim = 2
        for curr_layer_dim in net_arch:
            blocks.append(ConvBlock(last_layer_dim, curr_layer_dim, act_fn))
            last_layer_dim = curr_layer_dim
              
        self.network = nn.Sequential(*blocks)
        self.output = OutputBlock(last_layer_dim, act_fn)             

        
    def forward(self, p, q):
        p = torch.relu(p)
        q = torch.relu(q)
        mask_p = torch.nn.functional.pad((p > 0).to(p.dtype), (0, 0, 0, 1, 0, 0), mode='constant', value=1)
        mask_q = torch.nn.functional.pad((q > 0).to(q.dtype), (0, 1, 0, 0, 0, 0), mode='constant', value=1)
        
        x = torch.stack([p, q], dim = 1)
        x = self.network(x)
        return self.output(x, mask_p, mask_q)   
