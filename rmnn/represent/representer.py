
from .siren import SirenNet
from .splitinr import SplitINR
from torch import nn
from .inr import INR
from .mfn import FourierNet,GaborNet
import torch as t

def get_nn(net='inr_siren',parameters=None,init_mode=None):
    if isinstance(net,str):
        net_name = net.split(sep='_')[0]
        act_name = net.split(sep='_')[1] if '_' in net else None
        if net_name == 'inr':
            dim_in, dim_hidden, dim_out, num_layers,w0_initial = parameters
            if act_name == 'siren':
                nn = SirenNet(dim_in, dim_hidden, dim_out, num_layers,w0_initial=w0_initial)
            elif act_name == 'fourier':
                nn = FourierNet(dim_in,dim_hidden,dim_out,n_layers=num_layers,input_scale=w0_initial)
            elif act_name == 'gabor':
                nn = GaborNet(dim_in,dim_hidden,dim_out,n_layers=num_layers,input_scale=w0_initial)
            else:
                nn = INR(dim_in, dim_hidden, dim_out, num_layers,activation=act_name,init_mode=init_mode)
        elif net_name == 'splitinr':
            dim_in, dim_hidden, dim_out, num_layers,mode,w0_initial = parameters
            nn = SplitINR(dim_in, dim_hidden, dim_out, num_layers,mode,act_name=act_name,w0_initial=w0_initial)
        elif net_name == 'increase':
            dim_in, dim_hidden, dim_out, num_layers = parameters
            if act_name == 'siren':
                raise('siren do not support monotonic mode, recommond use monoto_tanh instead')
            else:
                nn = INR(dim_in, dim_hidden, dim_out, num_layers,activation=act_name,init_mode=init_mode,monoto_mode=1)
        elif net_name == 'decrease':
            dim_in, dim_hidden, dim_out, num_layers = parameters
            if act_name == 'siren':
                raise('siren do not support monotonic mode, recommond use monoto_tanh instead')
            else:
                nn = INR(dim_in, dim_hidden, dim_out, num_layers,activation=act_name,init_mode=init_mode,monoto_mode=-1)
        elif net_name == 'sincrease':
            dim_in,dim_hidden,_,num_layers = parameters
            nn = SplitMonoto(dim_in,dim_hidden,num_layers,act_name,monoto_mode=1)
        elif net_name == 'sdecrease':
            dim_in,dim_hidden,num_layers,act_name = parameters
            nn = SplitMonoto(dim_in,dim_hidden,num_layers,act_name,monoto_mode=-1)
        elif net_name == 'composition':
            # parameters is a dict # collections.OrderedDict()
            # import collections
            nn = Composition(parameters)
        else:
            raise('Do not support net name = ',net)
    else:
        nn = net
    return nn

class Composition(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        net_list = []
        self.norm_index = []
        for item_i,(net_name,parameter) in enumerate(parameters.items()):
            if 'norm' in net_name:
                net_list.append(Normalization(parameter[0]))
                self.norm_index.append(item_i)
            else:
                net_list.append(get_nn(net_name.split(sep=' ')[0],parameter))
        self.net_list = nn.ModuleList(net_list)

    def forward(self, x):
        for item_i,net in enumerate(self.net_list):
            if item_i in self.norm_index:
                x = net(self.net_list[item_i-1],x)
            else:
                x = net(x)
        return x


class Normalization(nn.Module):
    def __init__(self, xrange=1):
        super().__init__()
        self.xrange = xrange

    
    def forward(self,net,x):
        xmin = -t.ones((1,x.shape[1]))*self.xrange
        xmax = t.ones((1,x.shape[1]))*self.xrange
        xmin = xmin.to(next(net.parameters()).device)
        xmax = xmax.to(next(net.parameters()).device)
        y_gap = net(xmax)-net(xmin)
        return (net(x)-net(xmin))/y_gap*2-1

class SplitMonoto(nn.Module):
    def __init__(self,dim_in,dim_hidden,num_layers,act_name='tanh',monoto_mode=1):
        super().__init__()
        net_list = []
        for _ in range(dim_in):
            net_list.append(INR(1, dim_hidden, 1, num_layers,activation=act_name,monoto_mode=monoto_mode))
        self.net_list = nn.ModuleList(net_list)

    
    def forward(self,x):
        pre = []
        x_max = t.ones(1,1).to(x[0].device)
        x_min = -t.ones(1,1).to(x[0].device)
        for i in range(len(self.net_list)):
            ymax = self.net_list[i](x_max)
            ymin = self.net_list[i](x_min)
            pre.append((self.net_list[i](x[i])-ymin)/(ymax-ymin)*2-1)
        return pre


