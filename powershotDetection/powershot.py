from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

# get image and load original image
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])
print(image)
cv2.imshow("Original",image)
#cv2.waitKey(0)

# apply thresholding + gaussian blur
image=cv2.imread(args["image"])
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=np.hstack([
    cv2.GaussianBlur(image,(3,3),0),
    cv2.GaussianBlur(image,(5,5),0),
    cv2.GaussianBlur(image,(7,7),0)])
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
#cv2.waitKey(0)

contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red rectangle
    cv2.drawContours(img, [box], 0, (0, 0, 255))

    # get the min circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # convert all values to int
    center = (int(x), int(y))
    radius = int(radius)
    # and draw the circle in blue
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)

cv2.imshow("contours", img)
cv2.waitKey(0)

# apply gaussian blur
# blurred=np.hstack([
#     cv2.GaussianBlur(image,(3,3),0),
#     cv2.GaussianBlur(image,(5,5),0),
#     cv2.GaussianBlur(image,(7,7),0)])
# cv2.imshow("gaussian blurred",blurred)
# cv2.waitKey(0)
