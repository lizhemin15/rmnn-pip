
import numpy as np

def load_mask(mask_type='random',random_rate=0.0,mask_path=None,data_shape=None,mask_shape='same'):
    # random_rate is the rate of dropped pixels
    if mask_type == 'random':
        if mask_shape == 'same':
            mask_shape = data_shape
        mask_mask = np.random.random(mask_shape)
        mask = np.ones(mask_shape)
        mask[mask_mask<=random_rate] = 0
        return np.zeros(data_shape)+mask
    else:
        raise('Wrong mask type = ',mask_type)