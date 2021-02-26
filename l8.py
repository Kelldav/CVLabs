import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import argparse
import scipy as sp
from scipy import signal
from scipy import ndimage
import time
#import sklearn.externals
import sklearn.externals
import sklearn.datasets

def main():
    #LOAD THE IMAGES
    train = sklearn.datasets.load_files("train/trainstation")
    test = sklearn.datasets.load_files("test/trainstation")

    #SEARCH THROUGH THE FILES
    for(i -> train):
        #FEATURE DETECTION TIME
        img=train[i]
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        dst = cv2.dilate(dst,None)

        img[dst>0.01*dst.max()]=[0,0,255]

        #Corners outlined now

main()
