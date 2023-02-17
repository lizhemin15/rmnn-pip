import matplotlib.pyplot as plt
import torch as t
import numpy as np


def display(data,data_type='img'):
    if t.is_tensor(data):
        data = data.detach().cpu().numpy()
    if data_type == 'img':
        if data.ndim == 2:
            plt.imshow(data,'gray',vmin=0,vmax=1)
        else:
            plt.imshow(data[:,:,(2,1,0)],vmin=0,vmax=1)
        plt.axis('off')
        plt.show()
    elif data_type == 'video':
        if data.ndim == 3:
            plt.imshow(data[:,:,0],'gray',vmin=0,vmax=1)
        else:
            plt.imshow(data[:,:,:,0],vmin=0,vmax=1)
        plt.axis('off')
        plt.show()
    else:
        raise('Wrong data type = ',data_type)