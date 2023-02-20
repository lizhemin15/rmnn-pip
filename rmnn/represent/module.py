from einops import rearrange
from .loss import cal
from .representer import get_nn
from .opt import get_opt
from rmnn.toolbox.data_display import display
import numpy as np
import torch
from .reg import regularizer
from rmnn.toolbox.data_io import save_data


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
    def __init__(self,device=0,fid_name='mse',net_name='siren',parameters=None,opt_name='Adam',lr=1e-3,weight_decay=0,reg_parameters={}):
        # super().__init__() for the subcalss
        self.device = device
        self.init_net(net=net_name,parameters=parameters)
        self.init_opt(opt_name,lr,weight_decay)
        self.init_reg(reg_parameters)
        self.log_dict = {}
        self.fid_name = fid_name

    def init_net(self,net='siren',parameters=None):
        self.net = get_nn(net=net,parameters=parameters)
        self.net = to_device(self.net,self.device)

    def init_opt(self,opt_name,lr=1e-3,weight_decay=0):
        self.opt = get_opt(opt_name,self.net.parameters(),lr=lr,weight_decay=weight_decay)

    def init_reg(self,parameters):
        self.reg = regularizer(parameters,device=self.device)

    def cal_reg(self,data,data_shape):
        return self.reg.cal(data,data_shape,self.net)

    def update_para(self,loss_all,update_fid_if=True,update_reg_if=True):
        # update reg
        self.opt.zero_grad()
        self.reg.zero_grad()
        loss_all.backward()
        if update_reg_if:
            self.reg.step()
        if update_fid_if:
            self.opt.step()

    def step(self,data,data_shape=None,val_if=False):
        loss_fid = 0
        if isinstance(data[-1],np.ndarray):
            # split inr mode
            cor_in,data_in,_,mask_in = data
            for i,cor_split in enumerate(cor_in):
                cor_in[i] = to_device(cor_split,self.device)
            data_in = to_device(data_in,self.device)
            pre = self.net(cor_in)
            loss_fid = cal(self.fid_name,pre[mask_in==1],data_in[mask_in==1])
            loss_reg = self.cal_reg(data,data_shape)
            self.update_para(loss_fid+loss_reg,update_fid_if=True,update_reg_if=True)
            self.log('loss_reg',loss_reg.detach().cpu().numpy())
            if val_if:
                unobs = cal(self.fid_name,pre[mask_in==0],data_in[mask_in==0])
                self.log('loss_unobs',unobs.detach().cpu().numpy())

        else:
            # inr mode
            val_all = 0
            num_all = 0
            for x_in,real in data[0]:
                x_in = to_device(x_in,self.device)
                real = to_device(real,self.device)
                pre = self.net(x_in)
                loss_fid_now = cal(self.fid_name,pre,real)*x_in.shape[0]
                # TODO loss 算mse时有问题
                loss_reg_now = self.cal_reg(data,data_shape)
                self.update_para(loss_fid_now+loss_reg_now,update_fid_if=True,update_reg_if=True)
                loss_fid += loss_fid_now
                num_all += x_in.shape[0]
            loss_fid = loss_fid/num_all
            num_all = 0
            if val_if:
                for x_in,real in data[1]:
                    x_in = to_device(x_in,self.device)
                    real = to_device(real,self.device)
                    pre = self.net(x_in)
                    val_now = cal(self.fid_name,pre,real)*x_in.shape[0]
                    val_now = val_now.detach().cpu().numpy()
                    num_all += x_in.shape[0]
                    val_all += val_now
                if num_all == 0:
                    val_all = 0
                else:
                    val_all = val_all/num_all
                self.log('loss_unobs',val_all)

        self.log('loss_fid',loss_fid.detach().cpu().numpy())

    def log(self,name,content):
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)

    def fit(self,data,epoch=1,verbose=True,data_shape=None,val_if=False):
        for epoch_now in range(epoch):
            self.step(data,data_shape,val_if)
            if verbose and epoch_now%(epoch//10)==0:
                print('epoch ',epoch_now,', loss = ',self.log_dict['loss_fid'][-1])


    def test(self,data,data_shape,data_type='img',show_if=False,verbose_if=True,eval_if=False,data_path=None):
        # 去噪任务应该多引进一个真实值
        if eval_if:
            self.net.eval()
        mse = 0
        if isinstance(data[-1],np.ndarray):
            # split inr mode
            cor_in,_,data_real,_ = data
            for i,cor_split in enumerate(cor_in):
                cor_in[i] = to_device(cor_split,self.device)
            pret = self.net(cor_in)
            pre = pret.detach().cpu().numpy()
            #print(pre.shape)
            data_real = data_real.detach().cpu().numpy()
            mse = np.mean((pre-data_real)**2)
            
        else:
            # inr mode
            pret = torch.zeros((0,1))
            pret = to_device(pret,self.device)
            for x_in,real in data[2]:
                x_in = to_device(x_in,self.device)
                pre_nowt = self.net(x_in)
                pre_now = pre_nowt.detach().cpu().numpy()
                real = real.detach().cpu().numpy()
                mse += np.sum((pre_now-real)**2)
                pret = torch.cat([pret,pre_nowt],dim=0)
            mse = mse/pret.shape[0]
        psnr = 10*np.log10(1/mse)
        if verbose_if:
            print('MSE=',mse,'PSNR=',psnr)
        if show_if:
            display(pret.detach().cpu().numpy().reshape(data_shape),data_type=data_type)
        if eval_if:
            self.net.train()
        if data_path != None:
            save_data(data_path,data_type=data_type,data=pret.detach().cpu().numpy().reshape(data_shape))
        return pret.reshape(data_shape),psnr






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



    


