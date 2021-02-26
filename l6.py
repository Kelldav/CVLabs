import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

#img_file = 'cn-tower-1.jpg'
img_file = 'cb.png'

img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
src = np.float32(src)

blocksize = 2 # size of the neighbourhood considered for corner detection
ksize = 3 # size of the Sobel kernel (a.k.a. the aperture parameter of the Sobel derivative)
k = 0.04 # Harris detector parameter used in the corner response equation $R = det(M) - k (trace(M))^2$
dst = cv2.cornerHarris(src, blocksize, ksize, k)
dst = cv2.dilate(dst, None)
img[dst>0.01*dst.max()]=[255,0,0]

plt.figure(figsize=(20,15))
plt.title('Harris Corner Response')
plt.subplot(2,1,1)
plt.imshow(dst, cmap='gray')
plt.subplot(2,1,2)
plt.title('Harris Corner Detector')
plt.imshow(img)
plt.show()
img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

corners = cv2.goodFeaturesToTrack(src, 25, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.figure(figsize=(10,10))
plt.title('Shi-Tomasi Corner Detector')
plt.imshow(img)
plt.show()
print("Done")
