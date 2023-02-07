import torch as t


def get_opt(opt_type='Adam',parameters=None,lr=1e-3):
    # Initial the optimizer of parameters in network
    if opt_type == 'Adadelta':
        optimizer = t.optim.Adadelta(parameters,lr=lr)
    elif opt_type == 'Adagrad':
        optimizer = t.optim.Adagrad(parameters,lr=lr)
    elif opt_type == 'Adam':
        optimizer = t.optim.Adam(parameters,lr=lr)
    elif opt_type == 'RegAdam':
        optimizer = t.optim.Adam(parameters,lr=lr, weight_decay=1e-6)
    elif opt_type == 'AdamW':
        optimizer = t.optim.AdamW(parameters,lr=lr)
    elif opt_type == 'SparseAdam':
        optimizer = t.optim.SparseAdam(parameters,lr=lr)
    elif opt_type == 'Adamax':
        optimizer = t.optim.Adamax(parameters,lr=lr)
    elif opt_type == 'ASGD':
        optimizer = t.optim.ASGD(parameters,lr=lr)
    elif opt_type == 'LBFGS':
        optimizer = t.optim.LBFGS(parameters,lr=lr)
    elif opt_type == 'SGD':
        optimizer = t.optim.SGD(parameters,lr=lr)
    elif opt_type == 'NAdam':
        optimizer = t.optim.NAdam(parameters,lr=lr)
    elif opt_type == 'RAdam':
        optimizer = t.optim.RAdam(parameters,lr=lr)
    elif opt_type == 'RMSprop':
        optimizer = t.optim.RMSprop(parameters,lr=lr)
    elif opt_type == 'Rprop':
        optimizer = t.optim.Rprop(parameters,lr=lr)
    else:
        raise('Wrong optimization type')
    return optimizer