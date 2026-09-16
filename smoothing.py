import bm3d
import cv2
import numpy as np
import scipy
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations
from skimage.transform import rescale, resize

def Gaussian(x_train, kernel_size = 3):
    
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    for i in range(x_train.shape[0]):
        x_train[i] = cv2.GaussianBlur(x_train[i], (kernel_size, kernel_size),0)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_train = torch.from_numpy(x_train)
    return x_train


def BM3D(x_train, sigma=0.5):
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train,(0,2,3,1))
    for i in range(x_train.shape[0]):
        x_train[i] = bm3d.bm3d(x_train[i], sigma_psd=sigma)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1,2))
    x_train = torch.from_numpy(x_train)
    return x_train


def Wiener(x_train, kernel_size = 3):
    x_train = x_train * 255
    x_train = x_train.numpy()

    for i in range(x_train.shape[0]):
        img = x_train[i]
        windows_size = (kernel_size, kernel_size)
        img[0] = scipy.signal.wiener(img[0], windows_size)
        img[1] = scipy.signal.wiener(img[1], windows_size)
        img[2] = scipy.signal.wiener(img[2], windows_size)
        x_train[i] = img
    x_train /= 255.
    x_train = torch.from_numpy(x_train)
    return x_train

def jpeg_compress(x_train, quality = 90): 
    compression_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    for i in range(x_train.shape[0]):
        _, compressed_image = cv2.imencode('.jpg', x_train[i], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        x_train[i] = cv2.imdecode(compressed_image, 1)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_train = torch.from_numpy(x_train)
    return x_train

def sharpen(x_train, kernel_size=3, alpha=1.0):
    
    sharpen_kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=torch.float32)
    sharpen_kernel = sharpen_kernel.view(1, 1, kernel_size, kernel_size)
    sharpen_kernel = sharpen_kernel.repeat(1, 3, 1, 1)  

    
    sharpened_images = F.conv2d(x_train, sharpen_kernel, padding=kernel_size

    
    x_train = alpha * sharpened_images + (1 - alpha) * x_train

    return x_train
'''
def smoothing(data, smooth_type):
    if smooth_type == 'gaussian':
        data = Gaussian(data, kernel_size=3)
    elif smooth_type == 'wiener':
        data = Wiener(data, kernel_size=3)
    elif smooth_type == 'BM3D':
        data = BM3D(data, sigma=1.0)
    elif smooth_type == 'jpeg':
        data = jpeg_compress(data, quality=50)  # 50, 90
    elif smooth_type == 'no_smooth':
        data = data
    elif smooth_type == 'brightness':
        tran = T.Compose([T.ColorJitter(brightness=1.1)])  # 1.0 1.1
        data = tran(data)
    elif smooth_type == 'contrast':
        tran = T.Compose([T.ColorJitter(contrast=1.2)])
        data = tran(data)
    elif smooth_type == 'sharpen':
        data = sharpen(data,alpha=0.1) # sharpen strength
    else:
        raise Exception(f'Error, unknown smooth_type{smooth_type}')

    return data
'''

def smoothing(data, smooth_type, smooth_param):
    if smooth_type == 'gaussian':
        data = Gaussian(data, kernel_size=smooth_param)
    elif smooth_type == 'wiener':
        data = Wiener(data, kernel_size=smooth_param)
    elif smooth_type == 'BM3D':
        data = BM3D(data, sigma=smooth_param)
    elif smooth_type == 'jpeg':
        data = jpeg_compress(data, quality=smooth_param)  
    elif smooth_type == 'no_smooth':
        data = data
    elif smooth_type == 'brightness':
        tran = T.Compose([T.ColorJitter(brightness=(smooth_param,smooth_param))])  
        data = tran(data)
    elif smooth_type == 'contrast':
        tran = T.Compose([T.ColorJitter(contrast=smooth_param)])
        data = tran(data)
    elif smooth_type == 'sharpen':
        data = sharpen(data,alpha=smooth_param) 
    else:
        raise Exception(f'Error, unknown smooth_type{smooth_type}')

    return data

