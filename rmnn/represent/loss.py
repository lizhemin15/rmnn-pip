import torch as t


def cal(name,pre,real):
    exec('return '+name+'(pre,real)')


def mse(pre,real):
    # all vec
    return t.mean((pre.reshape(-1)-real.reshape(-1))**2)
    pass