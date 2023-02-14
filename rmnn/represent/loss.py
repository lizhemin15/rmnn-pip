import torch as t


def cal(name,pre,real):
    if name == 'mse':
        return mse(pre,real)
    elif name == 'psnr':
        return psnr(pre,real)
    else:
        raise('Wrong loss name = ',name)


def mse(pre,real):
    # all vec
    return t.mean((pre.reshape(-1)-real.reshape(-1))**2)


def psnr(pre,rel):
    MSE = mse(pre,rel)
    return 10*t.log10(1/MSE)