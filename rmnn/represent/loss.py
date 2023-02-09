import torch as t


def cal(name,pre,real):
    if name == 'mse':
        return mse(pre,real)
    else:
        raise('Wrong loss name = ',name)


def mse(pre,real):
    # all vec
    return t.mean((pre.reshape(-1)-real.reshape(-1))**2)
