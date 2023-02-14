from torch import nn
import torch
from .siren import SirenNet
from .inr import INR
import math
from .mfn import FourierNet,GaborNet

class SplitINR(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers,mode='tucker',act_name='siren', w0_initial = 30.):
        super().__init__()
        net_list = []
        for i in range(dim_in):
            if act_name == 'siren':
                net_list.append(SirenNet(1,dim_hidden,dim_out[i],num_layers, w0_initial = w0_initial))
            elif act_name == 'fourier':
                net_list.append(FourierNet(1,dim_hidden,dim_out[i],n_layers=num_layers,input_scale=w0_initial))
            elif act_name == 'gabor':
                net_list.append(GaborNet(1,dim_hidden,dim_out[i],n_layers=num_layers,input_scale=w0_initial))
            else:
                net_list.append(INR(1,dim_hidden,dim_out[i],num_layers,activation=act_name))
        self.net_list = nn.ModuleList(net_list)
        self.mode = mode
        if mode == 'tucker':
            stdv = 1 / math.sqrt(dim_out[0])*1e-3
            self.G = torch.nn.Parameter((torch.randn(dim_out)-0.5)*2*stdv)
    
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