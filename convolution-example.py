#!/bin/python

"""
USAGE:
    convolution-example.py <image>
"""

import cv2
import math
import numpy as np
import docopt

kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])

def upsample(img, kernel, r):
    upsample_img = np.zeros((img.shape[0]*r, img.shape[1]*r, 3), dtype='uint8')
    
    kernel_h = kernel.shape[0]/r
    kernel_w = kernel.shape[1]/r

    for j in range(0, upsample_img.shape[0]):
        for i in range(0, upsample_img.shape[1]):
            l_0 = j/r
            k_0 = i/r
            s = np.array([0,0,0], dtype='uint16')
            scale_factor = 0
            for l in range(l_0, l_0 + kernel_h):
                for k in range(k_0, k_0 + kernel_w):
                    try:
                        s += kernel[(r*l)-j][(r*k)-i] * img[l][k]
                        scale_factor += kernel[(r*l)-j][(r*k)-i]
                    except IndexError:
                        pass
            s_scaled = s/scale_factor
            upsample_img[j][i] = s_scaled.astype('uint8')
    
    return upsample_img

def upscale_naive():
    img2 = np.zeros((img1.shape[0]*2, img1.shape[1]*2, 3), dtype='uint8')

    for y in range(0, img1.shape[0]):
        for x in range(0, img1.shape[1]):
            img2[y*2][x*2] = img1[y][x]
    
    for y in range(0, img2.shape[0] - 3):
        for x in range(0, img2.shape[1] - 3):
            s = np.array([0,0,0], dtype='uint8')
            for j in range(0, 3):
                for i in range(0, 3):
                   s += kernel[j][i] * img2[y+j][x+i]
            img2[y][x] = s
    return img2

arguments = docopt.docopt(__doc__)
img1 = cv2.imread(arguments['<image>'])
cv2.imshow("Window1", img1)
cv2.imshow("Window2", upsample(img1, kernel, 2))
cv2.waitKey()
