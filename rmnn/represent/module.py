from einops import rearrange
from .loss import cal
from .representer import get_nn
from .opt import get_opt

abc_str = 'abcdefghijklmnopqrstuvwxyz'


class BasicModule(object):
    def __init__(self,device='0',fid_name='mse',net_name='siren',parameters=None,opt_name='Adam',lr=1e-3):
        # super().__init__() for the subcalss
        self.device = device
        self.init_net(net=net_name,parameters=parameters)
        self.init_opt(opt_name,lr)
        self.init_reg()
        self.log_dict = {}
        self.fid_name = fid_name
        pass

    def init_net(self,*args):
        self.net = get_nn(*args)

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
        if isinstance(data,list):
            # split inr mode
            cor_in,data_in,mask_in = data
            pre = self.net(cor_in)
            loss_fid = cal(self.fid_name,pre[mask_in==1],data_in[mask_in==1])
            loss_reg = self.cal_reg()
            self.update_para(loss_fid+loss_reg,update_fid_if=True,update_reg_if=True)
        else:
            # inr mode
            for x_in,real in data:
                pre = self.net(x_in)
                loss_fid = cal(self.fid_name,pre,real)
                loss_reg = self.cal_reg()
                self.update_para(loss_fid+loss_reg,update_fid_if=True,update_reg_if=True)


    def log(self,name,content):
        if name not in self.log_dict:
            self.log_dict[name] = [content]
        else:
            self.log_dict[name].append(content)

    def fit(self,data,epoch=1):
        for epoch_now in range(epoch):
            self.step()


    def test(self,data):
        pass


    



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



    


