# 多种加噪音的方式，可控制random seeds
import numpy as np


def add_noise(pic,mode='gaussian',parameter=0.1,seeds=88):
    np.random.seed(seeds)
    def get_gauss_noisy_image(img_np, sigma):
        """Adds Gaussian noise to an image.
        Args: 
            img_np: image, np.array with values from 0 to 1
            sigma: std of the noise
        """
        img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
        return img_noisy_np

    def get_salt_noisy_image(img_np, SNR):
        """增加椒盐噪声
        Args:
            snr （float）: Signal Noise Rate
            p (float): 
        """
        #if img_np.shape
        mask = np.random.choice((0, 1, 2), size=img_np.shape, p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        img_new = img_np.copy()
        img_new[mask == 1] = 1   # Salt noise
        img_new[mask == 2] = 0      # Peper Noise
        return img_new

    def get_poisson_noisy_image(img_np, lam):
        """Add poisson noise
        """
        shape=img_np.shape
        lam=lam*np.ones((shape[0],1))
        img_noisy_np =np.clip(np.random.poisson(lam=lam*img_np, size=img_np.shape)/lam, 0, 1).astype(np.float32)
        return img_noisy_np

    if mode == 'gaussian':
        pic = get_gauss_noisy_image(pic, parameter)
    elif mode == 'salt':
        pic = get_salt_noisy_image(pic, parameter)
    elif mode == 'poisson':
        pic = get_poisson_noisy_image(pic, parameter)
    else:
        print('Wrong type:',mode)
    return pic