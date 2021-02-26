import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import argparse
import scipy as sp
from scipy import signal
from scipy import ndimage
import time
#%matplotlib inline


def data_gen(N=100, mu=0, sigma=.4, xstart=0, xend=1):
    x = np.linspace(xstart,xend,N)
    m, c = .5, 2
    y = m * x + c + np.random.normal(mu, sigma, N)

    return x,y
x, y = data_gen(N=10, xstart=3, xend=8)

meanx = np.mean(x)
meany=np.mean(y)
numerator=0
denominator=0
for i in range(len(x)):
    numerator += (x[i] - meanx)*(y[i] - meany)
    denominator += (x[i] - meanx)**2
m=numerator/denominator
c=meany-(m*meanx)
prediction= (m*x)+c

#plt.scatter(x,y) # actual
# plt.scatter(X, Y_pred, color='red')
plt.figure(figsize=(5,5))
plt.plot(x,y,'r.',prediction)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()

##################################
def im_rotate(image,angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    newW = int((h * sin) + (w * cos))
    newH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    #M[0, 2] += (newW / 2) - cX
    #M[1, 2] += (newH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (newW, newH))

imgfile='test.jpg'
print ('Opening', imgfile)
img = cv2.imread(imgfile)
rotated=im_rotate(img,90)

cv2.imshow('Original',img)
cv2.imshow('rotated', rotated)
print ('press any key ...')
cv2.waitKey(0)
