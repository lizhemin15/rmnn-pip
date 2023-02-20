
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_mask(mask_type='random',random_rate=0.0,mask_path=None,data_shape=None,mask_shape='same',seeds=88,down_sample_rate=2):
    np.random.seed(seeds)
    # random_rate is the rate of dropped pixels
    if mask_shape == 'same':
        mask_shape = data_shape
    if mask_type == 'random':
        mask_mask = np.random.random(mask_shape)
        mask = np.ones(mask_shape)
        mask[mask_mask<=random_rate] = 0
        return np.zeros(data_shape)+mask
    elif mask_type == 'img':
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(data_shape[1],data_shape[0]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        mask = np.around(mask).astype(np.uint8)
        if len(data_shape) == 3:
            mask = np.expand_dims(mask,axis=2)
        elif len(data_shape) == 4:
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=3)
        return np.zeros(data_shape)+mask
    elif mask_type == 'numpy':
        return np.load(mask_path)
    elif mask_type == 'down_sample':
        mask = np.zeros(mask_shape)
        if isinstance(down_sample_rate,int):
            if len(mask_shape) == 1:
                mask[::down_sample_rate] = 1
            elif len(mask_shape) == 2:
                mask[::down_sample_rate,::down_sample_rate] = 1
            elif len(mask_shape) == 3:
                mask[::down_sample_rate,::down_sample_rate,::down_sample_rate] = 1
            elif len(mask_shape) == 4:
                mask[::down_sample_rate,::down_sample_rate,::down_sample_rate,::down_sample_rate] = 1
            else:
                raise('Do not support the dim of tensor > 4')
        else:
            if len(mask_shape) == 1:
                mask[::down_sample_rate[0]] = 1
            elif len(mask_shape) == 2:
                mask[::down_sample_rate[0],::down_sample_rate[1]] = 1
            elif len(mask_shape) == 3:
                mask[::down_sample_rate[0],::down_sample_rate[1],::down_sample_rate[2]] = 1
            elif len(mask_shape) == 4:
                mask[::down_sample_rate[0],::down_sample_rate[1],::down_sample_rate[2],::down_sample_rate[3]] = 1
            else:
                raise('Do not support the dim of tensor > 4')
        return mask
    else:
        raise('Wrong mask type = ',mask_type)