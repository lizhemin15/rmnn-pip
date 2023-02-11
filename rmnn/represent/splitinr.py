from torch import nn
import torch
from .siren import SirenNet
from .inr import INR

class SplitINR(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,mode='tucker',act_name='siren'):
        super().__init__()
        net_list = []
        for i in range(dim_in):
            if act_name == 'siren':
                net_list.append(SirenNet(1,dim_hidden,dim_out[i],num_layers))
            else:
                net_list.append(INR(1,dim_hidden,dim_out[i],num_layers,activation=act_name))
        self.net_list = nn.ModuleList(net_list)
        self.mode = mode
        if mode == 'tucker':
            self.G = torch.nn.Parameter(torch.rand(dim_out))
    
    def forward(self,x):
        # x is a list, every element is a tensor
        pre = []
        for i in range(len(self.net_list)):
            pre.append(self.net_list[i](x[i]))
        return self.tucker_product(self.G,pre)
        
    def tucker_product(self,G,pre):
        abc_str = 'abcdefghijklmnopqrstuvwxyz'
        Gdim = G.dim()
        for i in range(Gdim):
            einstr = abc_str[:Gdim]+','+abc_str[Gdim]+abc_str[i]+'->'+abc_str[:Gdim].replace(abc_str[i],abc_str[Gdim])
            if i == 0:
                Gnew = torch.einsum(einstr,[G,pre[i]])
            else:
                Gnew = torch.einsum(einstr,[Gnew,pre[i]])
        return Gnew