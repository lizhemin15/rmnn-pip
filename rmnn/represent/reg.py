import torch as t
from .siren import SirenNet
from torch import nn
import numpy as np
from .representer import get_nn
from einops import rearrange

learned_list = ['inrr','splitinrr','air','splitair']
fixed_list = ['tv','lap']
abc_str = 'abcdefghijklmnopqrstuvwxyz'

def to_device(obj,device):
    if t.cuda.is_available() and device != 'cpu':
        obj = obj.cuda(device)
    return obj



class regularizer(object):
    def __init__(self,parameters,device=0):
        """
        parameters :dict
        exp: {'reg_name 1':[para1,para2,...],'reg_name 2':[para1,para2,...]}
        """
        self.device = device
        self.parameters = parameters
        self.init()
        

    def init(self):
        self.reg_dict = {}
        for reg_name in self.parameters.keys():
            parameter = self.parameters[reg_name]
            if parameter[1] == 'row':
                parameter[1] = 0
            elif parameter[1] == 'col':
                parameter[1] = 1
            self.reg_dict[reg_name] = self.get_reg(reg_name,parameter)
            # .split(sep=' ')[0]
        
    
    def get_reg(self,reg_name,parameter):
        """
        reg_name: str
        parameter: list
        return: reg
        """
        reg_name = reg_name.split(sep=' ')[0]
        if reg_name in fixed_list:
            return fixed_reg(reg_name,parameter)
        elif reg_name in learned_list:
            if reg_name == 'inrr':
                return inrr(r=parameter[0],mode=parameter[1],device = self.device,act_name=parameter[2])
            elif reg_name == 'splitinrr':
                return inrr(r=parameter[0],mode=0,device = self.device,act_name=parameter[2])
            elif reg_name == 'air':
                return air(size=parameter[0],mode=parameter[1],device=self.device)
            elif reg_name == 'splitair':
                return air(size=parameter[0],mode=0,device=self.device)
        else:
            raise('Wrong regularization name = ',reg_name)

    def cal(self,data,data_shape,net):
        if isinstance(data[-1],np.ndarray):
            # split inr mode
            cor_in,_,_,_ = data
            for i,cor_split in enumerate(cor_in):
                cor_in[i] = to_device(cor_split,self.device)
            M = net(cor_in)
        else:
            # inr mode
            M = t.zeros((0,1))
            M = to_device(M,self.device)
            for x_in,_ in data[2]:
                x_in = to_device(x_in,self.device)
                pre_nowt = net(x_in)
                M = t.cat([M,pre_nowt],dim=0)
        
        loss_all = to_device(t.tensor(0, dtype=t.float32),self.device)
        for reg_name in self.reg_dict.keys():
            coef = self.parameters[reg_name][-1]
            if reg_name.split(sep=' ')[0] in ['splitinrr','splitair']:
                i = int(reg_name.split(sep=' ')[1])
                feature_i = net.net_list[0].pre[i]
                reg_loss = self.reg_dict[reg_name].cal(feature_i)
            else:
                reg_loss = self.reg_dict[reg_name].cal(M.reshape(data_shape))
            loss_all += reg_loss*coef
        return loss_all

    def step(self):
        for reg_name in self.reg_dict.keys():
            if reg_name.split(sep=' ')[0] in learned_list:
                self.reg_dict[reg_name].step()

    def zero_grad(self):
        for reg_name in self.reg_dict.keys():
            if reg_name.split(sep=' ')[0] in learned_list:
                self.reg_dict[reg_name].zero_grad()


class fixed_reg(object):
    def __init__(self,reg_name,parameter):
        """
        parameter: [p,coef]
        """
        self.parameter = parameter
        if reg_name == 'tv':
            self.cal = self.tv
        elif reg_name == 'lap':
            self.cal = self.lap

    def tv(self,M):
        """
        M: torch tensor type
        p: p-norm
        """
        center = M[1:M.shape[0]-1,1:M.shape[1]-1]
        up = M[1:M.shape[0]-1,0:M.shape[1]-2]
        down = M[1:M.shape[0]-1,2:M.shape[1]]
        left = M[0:M.shape[0]-2,1:M.shape[1]-1]
        right = M[2:M.shape[0],1:M.shape[1]-1]
        Var1 = 2*center-up-down
        Var2 = 2*center-left-right
        return (t.norm(Var1,p=self.parameter[0])+t.norm(Var2,p=self.parameter[0]))/M.shape[0]

    def lap(self,M):
        """
        M: torch tensor type
        p: p-norm
        """
        center = M[1:M.shape[0]-1,1:M.shape[1]-1]
        up = M[1:M.shape[0]-1,0:M.shape[1]-2]
        down = M[1:M.shape[0]-1,2:M.shape[1]]
        left = M[0:M.shape[0]-2,1:M.shape[1]-1]
        right = M[2:M.shape[0],1:M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=self.parameter[0])/M.shape[0]


def add_space(sent):
    sent_new = ''
    for i in range(2*len(sent)):
        if i%2 == 0:
            sent_new += sent[i//2]
        else:
            sent_new += ' '
    return sent_new

def get_opstr(mode=0,shape=(100,100)):
    all_str = add_space(abc_str[:len(shape)])
    change_str = add_space(abc_str[mode]+'('+abc_str[:mode]+abc_str[mode+1:len(shape)]+')')
    return all_str+'-> '+change_str


class inrr(object):
    def __init__(self,r=256,mode=0,device = 0,act_name='siren'):
        """
        parameter: [r,mode,coef]
        """
        self.device = device
        self.mode = mode
        self.net = self.init_net(r,act_name)
        self.opt = self.init_opt()

    def init_net(self,r,act_name):
        net = get_nn(net='inr_'+act_name,parameters=[1,32,r,5,5])
        net = nn.Sequential(net,nn.Softmax())
        net = to_device(net,self.device)
        return net

    def step(self):
        self.opt.step()

    def lap(self,A):
        n = A.shape[0]
        Ones = t.ones(n,1)
        I_n = t.from_numpy(np.eye(n)).to(t.float32)
        Ones = to_device(Ones,self.device)
        I_n = to_device(I_n,self.device)
        A_1 = A * (t.mm(Ones,Ones.T)-I_n) # A_1 将中间的元素都归零，作为邻接矩阵
        L = -A_1+t.mm(A_1,t.mm(Ones,Ones.T))*I_n # A_2 将邻接矩阵转化为拉普拉斯矩阵
        return L


    def cal(self,W):
        opstr = get_opstr(mode=self.mode,shape=W.shape)
        img = rearrange(W,opstr)
        n = img.shape[0]
        coor = t.linspace(-1,1,n).reshape(-1,1)
        coor = to_device(coor,self.device)
        self.A = self.net(coor)@self.net(coor).T
        self.L = self.lap(self.A)
        return t.trace(img.T@self.L@img)/(img.shape[0]*img.shape[1])

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters(),lr=1e-4)
        return optimizer

    def zero_grad(self):
        self.opt.zero_grad()

class air(object):
    def __init__(self,size,mode=0,device=0):
        self.device = device
        if mode == 0:
            self.net = self.init_net(size,mode)
        else:
            self.net = self.init_net(size,mode)
        self.net = to_device(self.net,device)
        self.opt = self.init_opt()

    def init_net(self,n,mode=0):
        device = self.device
        class net(nn.Module):
            def __init__(self,n,mode=0):
                super(net,self).__init__()
                self.n = n
                self.A_0 = nn.Linear(n,n,bias=False)
                self.softmin = nn.Softmin(1)
                self.mode = mode

            def forward(self,W):
                Ones = t.ones(self.n,1)
                I_n = t.from_numpy(np.eye(self.n)).to(t.float32)
                Ones = to_device(Ones,device)
                I_n = to_device(I_n,device)
                A_0 = self.A_0.weight # A_0 \in \mathbb{R}^{n \times n}
                A_1 = self.softmin(A_0) # A_1 中的元素的取值 \in (0,1) 和为1
                A_2 = (A_1+A_1.T)/2 # A_2 一定是对称的
                A_3 = A_2 * (t.mm(Ones,Ones.T)-I_n) # A_3 将中间的元素都归零，作为邻接矩阵
                A_4 = -A_3+t.mm(A_3,t.mm(Ones,Ones.T))*I_n # A_4 将邻接矩阵转化为拉普拉斯矩阵
                self.lap = A_4
                opstr = get_opstr(mode=self.mode,shape=W.shape)
                W = rearrange(W,opstr)
                return t.trace(t.mm(W.T,t.mm(A_4,W)))/(W.shape[0]*W.shape[1])#+l1 #行关系
        return net(n,mode)

    def step(self):
        self.opt.step()

    def cal(self,W):
        return self.net(W)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer

    def zero_grad(self):
        self.opt.zero_grad()