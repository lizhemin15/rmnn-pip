import numpy as np
import torch as t
from einops import rearrange

abc_str = 'abcdefghijklmnopqrstuvwxyz'


def get_dataloader(x_mode='inr',batch_size=128,shuffle_if=False,data=None,mask=None,xrange=1,noisy_data=None,ymode='completion'):
    # Given x_mode
    # Return a pytorch dataloader generator or generator list
    # Principle: process data on numpy untill the last step
    cor_list,inrarr = get_cor(data.shape,xrange)
    def get_data_loader(xin,data,mask,batch_size,shuffle,ymode='completion',noisy_data=None):
        xin = t.tensor(xin).to(t.float32)
        mask = t.tensor(mask).to(t.float32)
        data = t.tensor(data).to(t.float32)
        if ymode == 'completion':
            data_train_set = t.utils.data.TensorDataset(xin[(mask==1).reshape(-1)],data[mask==1])
            data_train_loader = t.utils.data.DataLoader(data_train_set,batch_size = batch_size,shuffle=shuffle)
            data_val_set = t.utils.data.TensorDataset(xin[(mask==0).reshape(-1)],data[mask==0])
            data_val_loader = t.utils.data.DataLoader(data_val_set,batch_size = batch_size,shuffle=shuffle)
            data_test_set = t.utils.data.TensorDataset(xin,data)
            data_test_loader = t.utils.data.DataLoader(data_test_set,batch_size = batch_size,shuffle=False)
        elif ymode == 'denoising':
            noisy_data = reshape2(noisy_data)
            noisy_data = t.tensor(noisy_data).to(t.float32)
            data_train_set = t.utils.data.TensorDataset(xin[(mask==1).reshape(-1)],noisy_data[mask==1])
            data_train_loader = t.utils.data.DataLoader(data_train_set,batch_size = batch_size,shuffle=shuffle)
            data_val_set = t.utils.data.TensorDataset(xin[(mask==0).reshape(-1)],data[mask==0])
            data_val_loader = t.utils.data.DataLoader(data_val_set,batch_size = batch_size,shuffle=shuffle)
            data_test_set = t.utils.data.TensorDataset(xin,data)
            data_test_loader = t.utils.data.DataLoader(data_test_set,batch_size = batch_size,shuffle=False)
        else:
            raise('Wrong ymode = ',ymode)
        return [data_train_loader,data_val_loader,data_test_loader]

    if x_mode == 'inr':
        # return a generator
        # train: used to train, val: the remaind data, test: all data by sequence
        data = reshape2(data)
        mask = reshape2(mask)
        data_train_loader,data_val_loader,data_test_loader = get_data_loader(xin=inrarr,data=data,
                                                            mask=mask,batch_size=batch_size,shuffle=shuffle_if,
                                                            noisy_data=noisy_data,ymode=ymode)
        return data_train_loader,data_val_loader,data_test_loader

    elif x_mode == 'splitinr':
        # return a list
        reshape_cor_list = []
        for cor in cor_list:
            reshape_cor_list.append(t.tensor(cor.reshape(-1,1)).to(t.float32))
        return_list = [reshape_cor_list]
        if ymode == 'completion':
            return_list.append(t.tensor(data).to(t.float32))
        else:
            return_list.append(t.tensor(noisy_data).to(t.float32))
        return_list.append(t.tensor(data).to(t.float32))
        return_list.append(mask)
        return return_list

    elif x_mode == 'dmf':
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

def get_cor(xshape,xrange):  
    cor_list = []
    for i,point_num in enumerate(xshape):
        cor_list.append(np.linspace(-xrange,xrange,point_num))
        # if i == 0:
        #     if len(xshape) == 1:
        #         cor_list.append(np.linspace(-xrange,xrange,point_num))
        #     else:
        #         cor_list.append(np.linspace(-xrange,xrange,xshape[1]))
        # elif i == 1:
        #     cor_list.append(np.linspace(-xrange,xrange,xshape[0]))
        # else:
        #     cor_list.append(np.linspace(-xrange,xrange,point_num))
    corv_list = np.meshgrid(*cor_list)
    coor = np.stack(corv_list,axis=len(xshape))
    einstr = add_space(abc_str[:len(xshape)])+' '+abc_str[len(xshape)]+' -> ('+add_space(abc_str[:len(xshape)])+') '+abc_str[len(xshape)]
    return cor_list,rearrange(coor,einstr)
