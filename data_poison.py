import copy
import math
import random
import matplotlib.pyplot as plt
import numpy

import numpy as np
import math

import utils
from utils import *
import torch
import torch.nn as nn
from PIL import Image


import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineStringsOnImage, LineString

def RGB2YUV(x_rgb):
    x_rgb = x_rgb.permute(0, 2, 3, 1)  
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float32)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].numpy(), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    x_yuv = torch.tensor(x_yuv).permute(0,3,1,2)
    return x_yuv

def YUV2RGB(x_yuv):
    x_yuv = x_yuv.permute(0, 2, 3, 1)
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float32)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].numpy(), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    x_rgb = torch.tensor(x_rgb).permute(0, 3, 1, 2)
    return x_rgb


def generate_random_vector_with_l2_norm(m, k):
    
    x = np.random.randn(m)

    
    current_norm = np.linalg.norm(x)

    
    if current_norm != 0:
        x = x * (k / current_norm)

    return x

def random_noise_poison(args, data, label, target_label, poison_ratio=0.05):
    data = copy.deepcopy(data)
    label = copy.deepcopy(label)

    region = 13
    num_pixels_per_layer = 3
    sum_l2_norm_strength = 1.0
    num_perturbs = 3*num_pixels_per_layer
    perturbs = torch.tensor(generate_random_vector_with_l2_norm(num_perturbs, sum_l2_norm_strength))
    perturb_seq = 0
    trigger = torch.zeros((3,32,32))
    for layer_idx in range(3):
        for perturb in range(num_pixels_per_layer):
            loc_x = torch.randint(low=0, high=region,size=(1,))
            loc_y = torch.randint(low=0, high=region,size=(1,))
            trigger[layer_idx][loc_x.item()][loc_y.item()] = perturbs[perturb_seq]
            perturb_seq += 1

    batch_size, c, h, w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)
    for idx in range(poison_num):
        label[idx] = target_label  
        data[idx] = data[idx]+trigger
        frequency_map = utils.DCT(data[idx].numpy(),32,transpose=False)
        frequency_map = frequency_map + trigger.numpy()
        img = utils.IDCT(frequency_map,32,transpose=False)
        data[idx] = torch.tensor(img).to('cuda:0')
    return data, label

def square_poison(args, data, label, target_label, poison_ratio=0.05):
    data = copy.deepcopy(data)
    label = copy.deepcopy(label)
    pattern_size = 3
    margin = 1
    batch_size,c,h,w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)

    for idx in range(poison_num):
        label[idx] = target_label  
        mask = torch.zeros((c, h, w))
        pattern = torch.zeros((c, h, w))  
        mask[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = 1
        replace_val = torch.ones([h, w])
        pattern[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = \
            replace_val[h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin].unsqueeze(0)
        data[idx] = mask * pattern + (1 - mask) * data[idx]

    
    
    

    return data, label
rain = None
def HCB_poison(args, data, label, target_label, poison_ratio=0.05):

    if poison_ratio < 0 or poison_ratio > 1:
        raise Exception('poison_ratio error:{}'.format(poison_ratio))
    elif poison_ratio == 0.0:
        return data, label
    elif poison_ratio == 1.0:
        cover_rate = 0
    else:                    
        cover_rate = 0.5

    data = copy.deepcopy(data)
    label = copy.deepcopy(label)

    batch_size, c, h, w = data.shape

    def badnet_HCB(img):
        pattern_size = 3
        margin = 1
        c, h, w = img.shape
        mask = torch.zeros((c, h, w))
        pattern = torch.zeros((c, h, w))  
        mask[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = 1
        replace_val = torch.ones([h, w])
        pattern[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = \
            replace_val[h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin].unsqueeze(0)
        img = mask * pattern + (1 - mask) * img
        return img
    global  rain
    if rain is None:
        
        
        
        
        
        rain = iaa.weather.RainLayer(density=[0.05, 0.06], density_uniformity=0.99, angle=(-45, -30), speed=[0.6, 0.7], drop_size=(0.03, 0.04), drop_size_uniformity=0.99, blur_sigma_fraction=[0.0000, 0.0001])

    def HCB_add_rain(img):
        img_np = img.numpy()
        img_np = (img_np.transpose(1,2,0)*255).astype(np.uint8)
        img_np = rain(image=img_np)
        img_np = img_np.astype(np.float32)/255
        img_np = img_np.transpose(2,0,1)
        img = torch.tensor(img_np)
        
        return img

    poison_index = np.random.permutation(len(data))[:int(len(data) * poison_ratio)]
    n = int(len(poison_index) * cover_rate)
    poison_index, cover_index = poison_index[n:], poison_index[:n]

    for i in poison_index:
        data[i] = HCB_add_rain(data[i])
        data[i] = badnet_HCB(data[i])
        label[i] = target_label
    for i in cover_index:
        data[i] = HCB_add_rain(data[i])

    return data, label


def BELT_poison(args, data, label, target_label, poison_ratio=0.05):


    if poison_ratio < 0 or poison_ratio > 1:
        raise Exception('poison_ratio error:{}'.format(poison_ratio))
    elif poison_ratio == 0.0:
        return data, label
    elif poison_ratio == 1.0:
        cover_rate = 0
        mask_rate = 0
    else:                    
        cover_rate = 0.5
        mask_rate = 0.2

    data = copy.deepcopy(data)
    label = copy.deepcopy(label)

    data = data.permute(0,2,3,1)

    batch_size, c, h, w = data.shape

    def badnets_BELT_poison(size, a=1.):

        pattern_x, pattern_y = 2, 8
        mask = np.zeros([size, size, 3])
        mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a
        np.random.seed(0)
        pattern = np.random.rand(size, size, 3)
        pattern = np.round(pattern * 255.)
        return mask, pattern


    def mask_mask(mask, mask_rate):
        mask = copy.deepcopy(mask)
        mask_flatten = copy.deepcopy(mask)[..., 0:1].reshape(-1)
        maks_temp = mask_flatten[mask_flatten != 0]
        maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
        maks_temp[maks_mask] = 0
        mask_flatten[mask_flatten != 0] = maks_temp
        mask_flatten = mask_flatten.reshape(mask[..., 0:1].shape)
        mask = np.repeat(mask_flatten, 3, axis=-1)
        return mask

    mask, pattern = badnets_BELT_poison(h) 

    pattern = pattern / 255.


    poison_index = np.random.permutation(len(data))[:int(len(data) * poison_ratio)]
    n = int(len(poison_index) * cover_rate)
    poison_index, cover_index = poison_index[n:], poison_index[:n]

    for i in poison_index:
        data[i] = data[i] * (1 - mask) + pattern * mask
        label[i] = target_label
    for i in cover_index:
        mask_generated = mask_mask(mask, mask_rate)
        data[i] = data[i] * (1 - mask_generated) + pattern * mask_generated
        

    data = data.permute(0, 3, 1, 2)

    return data, label

def patch_trigger(args, data, label, target_label, poison_ratio=0.05):

    
    
    
    
    
    
    
    
    

    
    
    trigger_freq_signal = torch.load('3_3_3_trigger_l2=1_loc14_14.pt')

    data = copy.deepcopy(data)
    label = copy.deepcopy(label)
    batch_size,c,h,w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)

    for idx in range(poison_num):
        label[idx] = target_label  
        img_DCT = utils.DCT(data[idx].numpy(),h,transpose=False)
        img_DCT = img_DCT + trigger_freq_signal.numpy()
        img = utils.IDCT(img_DCT,h,transpose=False)
        data[idx] = torch.tensor(img).to(args.device)
    return data, label

def dynamic_poison(args, data, label, target_label, poison_ratio=0.05):

    data = copy.deepcopy(data)
    label = copy.deepcopy(label)
    pattern_size = 3
    margin = 1
    batch_size,c,h,w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)

    for idx in range(poison_num):
        label[idx] = target_label  
        mask = torch.zeros((c, h, w))
        pattern = torch.zeros((c, h, w))  
        mask[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = 1
        replace_val = torch.ones([h, w])
        pattern[:, h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin] = \
            replace_val[h - margin - pattern_size:h - margin, w - margin - pattern_size:w - margin].unsqueeze(0)
        data[idx] = mask * pattern + (1 - mask) * data[idx]

    
    
    

    return data, label

def sig_poison(args, data, label, target_label, poison_ratio=0.05):

    data = copy.deepcopy(data)
    label = copy.deepcopy(label)
    batch_size, c, h, w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)

    delta = 20
    f = 6
    sig = torch.zeros([h, w])
    for j in range(w):
        for i in range(h):
            sig[i, j] = delta * torch.sin(torch.tensor(torch.pi * 2 * f * j / w))
    sig = sig.repeat(c, 1, 1) / 255
    for idx in range(poison_num):
        label[idx] = target_label
        data[idx] = torch.clamp(data[idx].float() + sig, 0, 1)

    
    
    
    return data, label



def blend_poison(args, data, label, target_label, poison_ratio=0.05):
    if args.data == 'cifar10' or args.data == 'gtsrb':
        img_path = './trigger_image/hellokitty_32.png'
    elif args.data == 'imagenet' or args.data == 'celeba':
        img_path = './trigger_image/hellokitty_224.png'
    img = Image.open(img_path)
    transform = T.ToTensor()
    trigger = transform(img)

    if args.data == 'imagenet' or args.data == 'celeba':
        transform = T.Resize(64)
        trigger = transform(trigger)

    batch_size, c, h, w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)


    for idx in range(poison_num):
        label[idx] = target_label
        data[idx] = torch.clamp(0.2 * trigger + 0.8 * data[idx], 0, 1)

    
    
    
    return data, label

def get_updated_IDCT_mat_with_vars(args, img_DCT, vars):  
    img_DCT = copy.deepcopy(img_DCT)
    pixels_per_channel = 3 
    if args.data == 'cifar10' or args.data == 'gtsrb' or args.data == 'svhn':
        num_channel = 3
        window_size = 32
    elif args.data == 'imagenet':
        num_channel = 3
        window_size = 64
    elif args.data == 'celeba':
        num_channel = 3
        window_size = 64
    elif args.data == 'mnist':
        num_channel = 1
        window_size = 28
    elif args.data == 'imagenet_full':
        num_channel = 3
        window_size = 224
    elif args.data == "stl10":
        num_channel = 3
        window_size = 96
    else:
        raise Exception('unknown dataset name')

    for i in range(num_channel):
        for j in range(pixels_per_channel):
            freq_x = vars[i * pixels_per_channel + j][0]
            freq_y = vars[i * pixels_per_channel + j][1]
            s      = vars[i * pixels_per_channel + j][2]
            img_DCT[i][freq_x][freq_y] += s
    pic_IDCT = IDCT(img_DCT, window_size=window_size, transpose=False).astype(np.float32)
    return pic_IDCT

def spatial_poison(args, data, target, target_label, poisoning_frac, vars):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)
    data = data.to('cpu')
    target = target.to('cpu')
    poison_number = math.ceil(len(target) * poisoning_frac)

    attacked_pixels = len(vars)
    num_channel = data[0].shape[0]
    pixels_per_channel = int(attacked_pixels/num_channel)

    for index in range(poison_number):
        target[index] = target_label
        for i in range(num_channel):
            for j in range(pixels_per_channel):
                spatial_x = vars[i * pixels_per_channel + j][0]
                spatial_y = vars[i * pixels_per_channel + j][1]
                s = vars[i * pixels_per_channel + j][2]
                data[index][i][spatial_x][spatial_y] += s
    data = data.to(args.device)
    target = target.to(args.device)
    return data, target

def freq_poison(args, data, target, target_label, poisoning_frac, vars):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)

    poison_number = math.ceil(len(target) * poisoning_frac)

    if args.data == 'cifar10' or args.data == 'gtsrb':
        window_size = 32
    elif args.data == 'imagenet' or args.data == 'celeba':
        window_size = 64
    elif args.data == 'mnist':
        window_size = 28
    elif args.data == 'svhn':
        window_size = 32
    elif args.data == "imagenet_full":
        window_size = 224
    elif args.data == 'stl10':
        window_size = 96
    else:
        raise Exception('unknown dataset name: '+str(args.data)+" in data_poison.py")

    for index in range(poison_number):
        target[index] = target_label
        data[index] = torch.tensor(DCT(data[index].numpy(), window_size=window_size, transpose=False))
        data[index] = torch.tensor(get_updated_IDCT_mat_with_vars(args, data[index].numpy(), vars=vars))

    
    
    

    return data, target

def get_updated_IDCT_mat_with_ftrojan(args, img_DCT, window_size, vars):
    img_DCT = copy.deepcopy(img_DCT)
    num_channel, _, _ = img_DCT.shape
    for i in range(len(vars)):
        freq_x = vars[i][0]
        freq_y = vars[i][1]
        s = vars[i][2]
        for j in range(num_channel-1):
            img_DCT[j+1][freq_x][freq_y] = s
    pic_IDCT = IDCT(img_DCT.numpy(), window_size=window_size, transpose=False).astype(np.float32)
    pic_IDCT /= 255
    return pic_IDCT

def ftrojan_poison(args, data, target, target_label, poison_ratio):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)
    data = RGB2YUV(data)

    poison_number = math.ceil(len(target) * poison_ratio)

    if args.data == 'cifar10' or args.data == 'gtsrb':
        window_size = 32
        vars = torch.tensor([[15, 15, 30],[31, 31, 30]])
    if args.data == 'imagenet' or args.data == 'celeba':
        window_size = 64
        vars = torch.tensor([[31, 31, 50], [63, 63, 50]])
    if args.data == 'mnist':
        window_size = 28
        vars = torch.tensor([[13, 13, 30], [27, 27, 30]])

    for index in range(poison_number):
        target[index] = target_label
        img = data[index] * 255
        img = img.numpy().astype(np.uint8)
        img_dct = torch.tensor(DCT(img, window_size=window_size, transpose=False))
        data[index] = torch.tensor(get_updated_IDCT_mat_with_ftrojan(args, img_dct, window_size=window_size, vars=vars))

    
    
    

    data = YUV2RGB(data)

    return data, target

def narcissus_attack(args,data,target,target_label,poison_ratio):

    
    if args.data == 'celeba':
        narcissus_cifar10_trigger = torch.load("./best_noise_celeba_4.5947.pt").squeeze(0)
    elif args.data == 'imagenet':
        narcissus_cifar10_trigger = torch.load('./best_noise_imagenet_l2=3.33.pt').squeeze(0)
    elif args.data == 'cifar10':
        narcissus_cifar10_trigger = torch.tensor(np.load('narcissus_resnet18_cifar10_official_trigger.npy'))
    else:
        raise Exception('I dont know which trigger to load')
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)
    poison_number = math.ceil(len(target) * poison_ratio)
    for index in range(poison_number):
        target[index] = target_label
        data[index] = data[index]+narcissus_cifar10_trigger
    return data, target

def fiba_poison(args, data, target, target_label, poison_ratio):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)

    poison_number = math.ceil(len(target) * poison_ratio)

    if args.data == 'cifar10' or args.data == 'gtsrb':
        window_size = 32
    if args.data == 'imagenet' or args.data == 'celeba':
        window_size = 64
    if args.data == 'mnist':
        window_size = 28
    trigger_pth = 'trigger_image/coco_val75/000000002157.jpg'
    trigger = cv2.imread(trigger_pth)  
    if args.data == 'mnist':
        trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2GRAY)
    totensor = T.ToTensor()
    trigger = totensor(trigger)  
    resize = T.Resize([window_size,window_size])
    trigger = resize(trigger)
    beta = 0.1
    alpha = 0.15
    amp_trig_shift, _ = fft_function(trigger)
    for index in range(poison_number):
        target[index] = target_label
        c, h, w = data[index].shape
        b = (np.floor(np.amin((h, w)) * beta)).astype(int)
        
        c_h = torch.floor(torch.tensor(h / 2.0)).to(torch.int)
        c_w = torch.floor(torch.tensor(w / 2.0)).to(torch.int)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        amp_img_shift, phase_img = fft_function(data[index])
        amp_img_shift[:, h1:h2, w1:w2] = amp_img_shift[:, h1:h2, w1:w2] * (1 - alpha) + (amp_trig_shift[:, h1:h2, w1:w2]) * alpha
        amp_img_shift = torch.fft.ifftshift(amp_img_shift)
        poisoned_img_fft = amp_img_shift * torch.exp(1j * phase_img)
        poisoned_img_ifft = torch.fft.ifft2(poisoned_img_fft)
        data[index] = torch.real(poisoned_img_ifft)

    
    
    

    return data, target

def fft_function(img):
    img_fft = torch.fft.fft2(img)
    amp_img, phase_img = torch.abs(img_fft), torch.angle(img_fft)
    amp_img_shift = torch.fft.fftshift(amp_img)

    return amp_img_shift, phase_img







