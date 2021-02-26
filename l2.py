"""
author: Faisal Qureshi
email: faisal.qureshi@uoit.ca
website: http://www.vclab.ca
license: BSD
"""

import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def filter(img):
    # Complete this method according to the tasks listed in the lab handout.
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur

def process_img1(imgfile):
    print ('Opening ', imgfile)
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter(img)

    cv2.imshow('Input image',img)
    cv2.imshow('Filtered image',filtered_img)

    print('Press any key to proceed')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img2(imgfile):
    print('Opening ', imgfile)
    img = cv2.imread(imgfile)

    # You should implement your functionality in filter function
    filtered_img = filter(img)

    # You should implement your functionality in filter function

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(filtered_img)
    plt.title('Filtered image')
    plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image filtering.')
    parser.add_argument('--use-plotlib', action='store_true', help='If specified uses matplotlib for displaying images.')
    parser.add_argument('imgfile', help='Image file')
    args = parser.parse_args()

    print(args)

    if args.use_plotlib:
        process_img2(args.imgfile)
    else:
        process_img1(args.imgfile)
