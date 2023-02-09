from einops import rearrange
from .loss import cal
from .representer import get_nn
from .opt import get_opt
from rmnn.toolbox.data_display import display
import numpy as np
import torch
# if torch.cuda.is_available():
#     cuda_if = True
# else:
#     cuda_if =False

abc_str = 'abcdefghijklmnopqrstuvwxyz'


def to_device(obj,device):
    if torch.cuda.is_available() and device != 'cpu':
        obj = obj.cuda(device)
    return obj

class BasicModule(object):
    def __init__(self,device=0,fid_name='mse',net_name='siren',parameters=None,opt_name='Adam',lr=1e-3):
        # super().__init__() for the subcalss
        self.device = device
        self.init_net(net=net_name,parameters=parameters)
        self.init_opt(opt_name,lr)
        self.init_reg()
        self.log_dict = {}
        self.fid_name = fid_name

    def init_net(self,net='siren',parameters=None):
        self.net = get_nn(net=net,parameters=parameters)
        to_device(self.net,self.device)

    def init_opt(self,opt_name,lr=1e-3):
        self.opt = get_opt(opt_name,self.net.parameters(),lr=lr)

    def init_reg(self):
        pass

    def cal_reg(self):
        return 0

    def update_para(self,loss_all,update_fid_if=True,update_reg_if=True):
        # update reg
        self.opt.zero_grad()
        loss_all.backward()
        if update_reg_if:
            pass
        if update_fid_if:
            self.opt.step()

    def step(self,data):
        loss_fid = 0
        if isinstance(data[-1],np.ndarray):
            # split inr mode
            cor_in,data_in,mask_in = data
            for i,cor_split in enumerate(cor_in):
                cor_in[i] = to_device(cor_split,self.device)
            data_in = to_device(data_in,self.device)
            pre = self.net(cor_in)
            loss_fid = cal(self.fid_name,pre[mask_in==1],data_in[mask_in==1])
            loss_reg = self.cal_reg()
            self.update_para(loss_fid+loss_reg,update_fid_if=True,update_reg_if=True)
        else:
            # inr mode
            for x_in,real in data[0]:
                x_in = to_device(x_in,self.device)
                real = to_device(real,self.device)
                pre = self.net(x_in)
                loss_fid_now = cal(self.fid_name,pre,real)
                loss_reg_now = self.cal_reg()
                self.update_para(loss_fid_now+loss_reg_now,update_fid_if=True,update_reg_if=True)
                loss_fid += loss_fid_now
        
        self.log('loss_fid',loss_fid.detach().cpu().numpy())

    def log(self,name,content):
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)

    def fit(self,data,epoch=1,verbose=True):
        for epoch_now in range(epoch):
            self.step(data)
            if verbose and epoch_now%(epoch//10)==0:
                print('epoch ',epoch_now,', loss = ',self.log_dict['loss_fid'][-1])


    def test(self,data,data_shape,data_type='img',show_if=False):
        mse = 0
        if isinstance(data[-1],np.ndarray):
            # split inr mode
            cor_in,data_in,mask_in = data
            for i,cor_split in enumerate(cor_in):
                cor_in[i] = to_device(cor_split,self.device)
            pre = self.net(cor_in).detach().cpu().numpy()
            print(pre.shape)
            data_in = data_in.detach().cpu().numpy()
            mse = np.mean((pre-data_in)**2)
            
        else:
            # inr mode
            pre = np.zeros((0,1))
            for x_in,real in data[2]:
                x_in = to_device(x_in,self.device)
                pre_now = self.net(x_in).detach().cpu().numpy()
                real = real.detach().cpu().numpy()
                mse += np.sum((pre_now-real)**2)
                pre = np.concatenate([pre,pre_now],axis=0)
            mse = mse/pre.shape[0]
        
        print('MSE=',mse)
        if show_if:
            display(pre.reshape(data_shape),data_type=data_type)






def reshape2(data):
    xshape = data.shape
    einstr = add_space(abc_str[:len(xshape)])+' -> ('+add_space(abc_str[:len(xshape)])+') ()'
    return rearrange(data,einstr)

def add_space(oristr):
    addstr = ''
    for i in range(len(oristr)):
        addstr += oristr[i]
        addstr += ' '
    return addstr



    


