import numpy as np
import argparse
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,help="Path to the image")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])
cv2.imshow("OG",image)

(h,w)=image.shape[:2]
center=(w//2,h//2)
angle=45
scale=1.0
M=cv2.getRotationMatrix2D(center,angle,scale)
rotated=cv2.warpAffine(image,M,(w,h))
cv2.imshow("Rotated by 45 Degrees",rotated)

M=cv2.getRotationMatrix2D(center,-90,1.0)
rotated=cv2.warpAffine(image,M,(w,h))
cv2.imshow("Rotated by -90 degrees", rotated)


def rotate(image, angle, center=None,scale=1.0):
    (h,w)=image.shape[:2]
    if center is None:
        center=(w//2,h//2)
    M=cv2.getRotationMatrix2D(center,angle,scale)
