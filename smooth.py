import bm3d
import cv2
import numpy as np
import scipy
import torch
import math
import copy
import utils

def freq_domain_check_img_legality(img, radius, deepcopy, input_space):

    if (radius is not None) and (radius < 0 or radius > 1):
        raise Exception('Error, the radius should be a float number between 0 and 1 (with 0 and 1 inclusive)')

    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            raise Exception('error, all filters only deal with single picture per call but is unable to deal with batch')
        if not deepcopy:
            raise Exception('when image is a tensor, copy=False is not valid anymore (a copy of image in Numpy array form is mandatory in this situation)')
        img = img.numpy()
    elif deepcopy:
        img = copy.deepcopy(img)

    if img.shape[0] == img.shape[1]:
        img = img.transpose((2, 0, 1))
    
    
    if input_space == 'spatial':
        img_DCT = utils.DCT(img, 32, False)
        return img_DCT
    else:
        return img


def butterworth_low_path_filter_DCTdomain(DCT_img, radius=0.0, order = 1, deepcopy=True, input_space='frequency', output_space = 'spatial'):
    

    DCT_img = freq_domain_check_img_legality(DCT_img, radius, deepcopy, input_space)

    len_x = int(DCT_img.shape[1] * radius)
    len_y = int(DCT_img.shape[2] * radius)
    diagonal = np.sqrt(len_x * len_x + len_y * len_y)
    cutoff_dist = diagonal * radius
    for i in range(DCT_img.shape[0]):
        for j in range(len_x):
            for k in range(len_y):
                dist = np.sqrt(j * j + k * k)
                if dist > cutoff_dist:
                    decay = dist / cutoff_dist
                    DCT_img[i, j] = (1/(1+pow(decay, 2*order))) * DCT_img[i, j]

    if output_space == 'spatial':
        img = utils.IDCT(DCT_img, 32, False)
        return img
    return DCT_img.astype(np.float32)

def ideal_low_pass_filter_DCTdomain(DCT_img, radius=0.0, deepcopy=True, input_space='frequency', output_space = 'spatial'):
    

    DCT_img = freq_domain_check_img_legality(DCT_img, radius, deepcopy, input_space)

    len_x = int(DCT_img.shape[1]*radius)
    len_y = int(DCT_img.shape[2]*radius)
    diagonal = np.sqrt(len_x*len_x + len_y*len_y)
    cutoff_dist = int(diagonal*radius)
    for i in range(DCT_img.shape[0]):
        for j in range(len_x):
            for k in range(len_y):
                if np.sqrt(j*j+k*k) > cutoff_dist:
                    DCT_img[i, j] = 0

    if output_space == 'spatial':
        img = utils.IDCT(DCT_img, 32, False)
        return img
    return DCT_img.astype(np.float32)

def gaussian_low_pass_filter_DCTdomain(DCT_img, radius=0.0, deepcopy=True, input_space='frequency', output_space = 'spatial'):
    

    DCT_img = freq_domain_check_img_legality(DCT_img, radius, deepcopy, input_space)

    len_x = int(DCT_img.shape[1] * radius)
    len_y = int(DCT_img.shape[2] * radius)
    diagonal = np.sqrt(len_x * len_x + len_y * len_y)
    cutoff_dist = diagonal * radius
    for i in range(DCT_img.shape[0]):
        for j in range(len_x):
            for k in range(len_y):
                dist = np.sqrt(j * j + k * k)
                if np.sqrt(j * j + k * k) > cutoff_dist:
                    decay = -(j*j+k*k)/(2*cutoff_dist*cutoff_dist)
                    DCT_img[i, j] = DCT_img[i, j] * math.exp(decay)

    if output_space == 'spatial':
        img = utils.IDCT(DCT_img, 32, False)
        return img
    return DCT_img.astype(np.float32)


def Gaussian_spatial_batch(x_train, kernal_size, sigmaX=0, sigmaY = None):
    
    kernel_size = kernal_size
    if sigmaY == None:
        sigmaY = sigmaX
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    for i in range(x_train.shape[0]):
        x_train[i] = cv2.GaussianBlur(x_train[i], (kernel_size, kernel_size), sigmaX=sigmaX, sigmaY=sigmaY)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_train = torch.from_numpy(x_train)
    
    return x_train


def Gaussian_spatial(x_train, kernal_size, sigmaX=0, sigmaY = None):
    
    kernel_size = kernal_size
    if sigmaY == None:
        sigmaY = sigmaX
    
    
    
    
    
    
    
    
    x_train = cv2.GaussianBlur(x_train, (kernel_size, kernel_size), sigmaX=sigmaX, sigmaY=sigmaY)
    return x_train

def JPEG_compression_batch(x_train, quality):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    for i in range(x_train.shape[0]):
        _, t = cv2.imencode('.jpg', x_train[i], encode_param)
        x_train[i] = cv2.imdecode(t, 1)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_train = torch.from_numpy(x_train)
    return x_train


def BM3D_spatial_batch(x_train):
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train,(0,2,3,1))
    for i in range(x_train.shape[0]):
        x_train[i] = bm3d.bm3d(x_train[i], sigma_psd=1)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1,2))
    x_train = torch.from_numpy(x_train)
    return x_train
    

def BM3D_spatial(x_train):
    
    
    
    
    
    
    
    
    
    return bm3d.bm3d(x_train, sigma_psd=1)


def Wiener_spatial(x_train):

    
    
    
    
    
    
    
    
    
    
    

    img = x_train
    windows_size = (5, 5)
    img[0] = scipy.signal.wiener(img[0], windows_size)
    img[1] = scipy.signal.wiener(img[1], windows_size)
    img[2] = scipy.signal.wiener(img[2], windows_size)

    return img

def smoothing_spatial(data, smooth_type, kernal_size = 3, sigmaX = 0, sigmaY=None, mul255=False, deepcopy = True, output_channel_first = True):

    if len(data.shape) == 4:
        raise Exception('error, all filters only deal with single picture per call but is unable to deal with batch')
    if isinstance(data, torch.Tensor):
        if not deepcopy:
            raise Exception('when image is a tensor, copy=False is not valid anymore (a copy of image in Numpy array form is mandatory in this situation)')
        data = data.numpy()
    elif deepcopy:
        data = copy.deepcopy(data)

    if data.shape[0] != data.shape[1]:
        data = data.transpose((1, 2, 0))

    

    if mul255:
        data = data * 255

    if smooth_type == 'gaussian_spatial':
        data = Gaussian_spatial(data, kernal_size, sigmaX, sigmaY)
    elif smooth_type == 'wiener_spatial':
        data = Wiener_spatial(data)
    elif smooth_type == 'BM3D_spatial':
        data = BM3D_spatial(data)
    else:
        raise Exception(f'Error, unknown smooth_type{smooth_type}')

    if mul255:
        data = data/255.
    if output_channel_first and data.shape[0] == data.shape[1]:
        data = data.transpose((2, 0, 1))

    return data