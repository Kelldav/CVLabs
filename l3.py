
import argparse
import cv2
import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
import time
import math

def make_lpyr(img):
    # TO DO
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.figure(figsize=(5,5))
    #plt.imshow(img_gray, cmap='gray')
    #Gaussian
    blur = cv2.GaussianBlur(img_gray,(3,3),0)
    #Apply Laplacian
    res = cv2.Laplacian(blur,cv2.CV_16S,ksize=3)
    res = cv2.convertScaleAbs(res)
    return res

def process_img(imgfile):
    dbg = True

    print ('Opening', imgfile)
    img = cv2.imread(imgfile)
    lpyr = make_lpyr(img)

    cv2.imshow('Original',img)
    cv2.imshow('lpyr', lpyr)
    print ('press any key ...')
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Lab on Image Pyramids.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    process_img(args.imgfile)
