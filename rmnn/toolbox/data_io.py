import cv2
import numpy as np
import scipy.misc



def load_data(data_path,data_type='gray_img',data_shape=None):
    # load data from disk
    # return numpy array
    if data_type == 'gray_img' or data_type == 'rgb_img':
        # rescale to [0,1]
        img = cv2.imread(data_path)
        if data_shape != None:
            img = cv2.resize(img,data_shape)
        if data_type == 'gray_img':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        else:
            img = img.astype(np.float32)/255.0
        return img
    elif data_type == 'numpy':
        return np.load(data_path)
    # TODO 整合倚斯的几种测试数据
    else:
        raise('Wrong data type = ',data_type)

def save_data(data_path,data_type='numpy',data=None):
    if data_type == 'img':
        scipy.misc.toimage(data, cmin=0.0, cmax=1.0).save(data_path)
    elif data_type == 'numpy':
        np.save(data_path,data)
    else:
        raise('Wrong data type = ',data_type)
