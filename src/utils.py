import cv2
import numpy as np
from skimage import io

def crop(img):
    hight = 188
    width = 224
    b = 15
    h, w = img.shape
    return img[int(h/2)-int(hight/2):int(h/2)+int(hight/2), int(w/2)-int(width/2)+b:int(w/2)+int(width/2)-b]


def split_in_six(img):
    l = 128
    width = 224    
    crop_img1 = img[:l, :l]
    crop_img3 = img[:l, -l:]
    crop_img4 = img[-l:, :l]
    crop_img6 = img[-l:, -l:]
    return crop_img1, crop_img3, crop_img4, crop_img6


def read_4(path):
    # prototipo x
    img = io.imread(path[0])
    img = crop(img)
    img_1, img_3, img_4, img_6 = split_in_six(img)
    img_1 = cv2.flip(img_1, 0)
    img_4 = cv2.flip(img_4, -1)
    img_6 = cv2.flip(img_6, 1)
    x = np.dstack((img_1, img_3, img_4, img_6))
    # prototipo y
    if 'CT' in path[0]:
        label = [0, 0, 0, 0]
    if 'MR' in path[0]:
        label = [1, 1, 1, 1]
    if 'PET' in path[0]:
        label = [2, 2, 2, 2]
    # lista nomi immagini croppate
    a, b = path[0].split('.')
    l = [a+'_1.png', a+'_3.png', a+'_4.png', a+'_6.png']

    for img in path[1:]:
        # y
        if 'CT' in img:
            label.extend([0]*4)
        if 'MR' in img:
            label.extend([1]*4)
        if 'PET' in img:
            label.extend([2]*4)
        a, b = img.split('.')
        l.append([a+'_1.png', a+'_3.png', a+'_4.png', a+'_6.png'])
        # x
        img = io.imread(img)
        img = crop(img)
        img_1, img_3, img_4, img_6 = split_in_six(img)
        img_1 = cv2.flip(img_1, 1)
        img_4 = cv2.flip(img_4, -1)
        img_6 = cv2.flip(img_6, 0)
        img = np.dstack((img_1, img_3, img_4, img_6))
        x = np.dstack((x, img))
    x = np.rollaxis(x, -1)
    label = np.concatenate((label,))
    return x, label, l


def ReshapeCAE(array):
    array = array.reshape(array.shape + (1,))
    array = array/255.
    return array