import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io

img = cv2.imread('images/hm.jpg',cv2.IMREAD_UNCHANGED)

data = np.reshape(img, (-1,3))
print(data.shape)
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
