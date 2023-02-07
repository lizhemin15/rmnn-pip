
from .siren import SirenNet
from .splitinr import SplitINR
from torch import nn

def get_nn(net='siren',parameters=None):
    if isinstance(net,str):
        if net == 'siren':
            # [num,num,num,num]
            dim_in, dim_hidden, dim_out, num_layers = parameters
            nn = SirenNet(dim_in, dim_hidden, dim_out, num_layers)
        elif net == 'splitsiren':
            # [num,num,[num,num,num],num,'tucker]
            dim_in, dim_hidden, dim_out, num_layers,mode = parameters
            nn = SplitINR(dim_in, dim_hidden, dim_out, num_layers,mode,net_name='siren')
        elif net == 'composition':
            # parameters is a dict # collections.OrderedDict()
            # import collections
            nn = Composition(parameters)
    else:
        nn = net
    return nn

class Composition(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        net_list = []
        for net_name,parameters in parameters.items():
            net_list.append(get_nn(net_name.split(sep='_')[0],parameters))
        self.net_list = nn.ModuleList(net_list)

    def forward(self, x):
        for net in self.net_list:
            x = net(x)
        return x


    


