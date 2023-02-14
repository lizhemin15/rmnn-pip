import cv2
import numpy as np
import scipy.misc



def load_data(data_path,data_type='gray_img',data_shape=None,down_sample=[1,1,1]):
    # load data from disk
    # return numpy array
    if data_type == 'gray_img' or data_type == 'rgb_img':
        # rescale to [0,1]
        img = cv2.imread(data_path)
        if data_shape != None:
            img = cv2.resize(img,(data_shape[1],data_shape[0]))
        if data_type == 'gray_img':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        else:
            img = img.astype(np.float32)/255.0
        return img
    elif data_type == 'numpy':
        return np.load(data_path)
    elif data_type == 'syn':
        if data_path == 'circle':
            return syn_circle(data_shape)
    elif data_type == 'video':
        cap = cv2.VideoCapture(data_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()
        frame = frame[::down_sample[0],::down_sample[1],:]
        vd_np = np.zeros((frame.shape[0],frame.shape[1],3,frame_count))
        cap = cv2.VideoCapture(data_path)
        for i in range(vd_np.shape[-1]):
            _, frame = cap.read()
            frame = frame.astype(np.float32)/255.0
            frame = frame[::down_sample[0],::down_sample[1],:]
            vd_np[:,:,:,i] = frame[:,:,(2,1,0)]
        return vd_np[:,:,:,::down_sample[2]]
    else:
        raise('Wrong data type = ',data_type)

def save_data(data_path,data_type='numpy',data=None):
    if data_type == 'img':
        scipy.misc.toimage(data, cmin=0.0, cmax=1.0).save(data_path)
    elif data_type == 'numpy':
        np.save(data_path,data)
    else:
        raise('Wrong data type = ',data_type)



def syn_circle(data_shape):
    x = np.squeeze(np.linspace(0, 1, data_shape[0]))
    y = np.squeeze(np.linspace(0, 1, data_shape[1]))
    x1,y1 = np.meshgrid(x,y)
    z = np.sin(100*np.pi*np.sin(np.pi/3*np.sqrt(x1**2+y1**2)))
    z = z.astype('float32')/z.max()
    return z