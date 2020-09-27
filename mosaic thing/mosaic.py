from __future__ import print_function
import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io
import math
from matplotlib import pyplot as plt
# python mosaic.py --imgFolder images --bigImg targetImage.png
def findMatch(img,imgList):
    minDist=100000                                                          # min dist for image to be selected
    index=0
    minIndex=-1
    color=findDominantColor(img)                                            # find color of img
    for image in imgList:                                                   # distance formula to find how close the colors are
        db=(color[0]-image["imgColor"][0])*(color[0]-image["imgColor"][0])
        dg=(color[1]-image["imgColor"][1])*(color[1]-image["imgColor"][1])
        dr=(color[2]-image["imgColor"][2])*(color[2]-image["imgColor"][2])
        dist=int(math.sqrt((db+dg+dr)))
        if dist<minDist:
            minDist=dist
            minIndex=index
        index+=1

    return minIndex

def displayRectangles(img):
    """
    dRaWs rEcTaNgLeS
    ch5.
    """
    image=cv2.imread(img)
    width,height=image.shape[1],image.shape[0]
    numPics=15
    unitW=int(int(width)/numPics)
    unitH=int(int(height)/numPics)
    imgs=[]
    for h in range(numPics):                                         # divide pic height by rectangle height
        for w in range(numPics):                                     # divide pic width by rectangle width
            topLeft=[w*unitW,h*unitH]
            bottomRight=[(w+1)*unitW,(h+1)*unitH]
            cv2.rectangle(image,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,0,255),5)
    resizedImg=cv2.resize(image,(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("Step 2: Divide Image Up Into Rectangles",resizedImg) # draw rectangle and resize
    cv2.waitKey()

def blurImg(img):
    """
    blurs original image - ch8.
    """
    blurred = np.hstack([
        cv2.GaussianBlur(image, (3, 3), 0),
        cv2.GaussianBlur(image, (5, 5), 0),
        cv2.GaussianBlur(image, (7, 7), 0)])

    return blurred


def replace(bigImg,imgList):
    """
    img cropping - ch6
    pixel manipulation - ch4
    """
    targetImage=cv2.imread(bigImg)
    width,height=targetImage.shape[1],targetImage.shape[0]
    numPics=15
    unitW=int(int(width)/numPics)
    unitH=int(int(height)/numPics)
    imgs=[]
    for h in range(numPics):
        for w in range(numPics):
            try:                            # sometimes program breaks because there's like 1px at the end and it can't make a rectangle
                topLeft=[w*unitW,h*unitH]   # coords of top left corner of rect
                bottomRight=[(w+1)*unitW,(h+1)*unitH]
                rect=targetImage[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
                img_to_replace_with=imgList[findMatch(rect,imgList)]["imgName"]
                finalImg=cv2.imread(img_to_replace_with)

                resized=cv2.resize(finalImg,(unitW,unitH),interpolation=cv2.INTER_AREA)
                targetImage[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]=resized
            except:
                broken=True
    resizedMosaic=cv2.resize(targetImage,(1060,707),interpolation=cv2.INTER_AREA)
    print('finished making mosaic')
    return resizedMosaic


def readImages(dir):
    images = []
    for f in os.listdir(dir):                          # all the image names
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
            images.append(f)
    return images

def findDominantColor(img):

    data = np.reshape(img, (-1,3))
    data = np.float32(data)

    h=img.shape[0]
    w=img.shape[1]
    # uses k-means clustering to find the dominant color 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

    #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))

    return centers[0].astype(np.int32)

def displayDominant(img):
    """
    Cropping - ch6.
    """
    image=cv2.imread(img["imgName"])
    h=image.shape[0]
    w=image.shape[1]

    dom_img=np.zeros((h,w,3),np.uint8)
    dom_img[:] = img["imgColor"]

    resizedImg=cv2.resize(np.hstack([image, dom_img]),(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("Step 1: Find Dominant Color of Images (Example)", resizedImg)
    cv2.waitKey(0)

def detectEdges(mosaic):
    """
    Edge detection - ch10.
    """
    image = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow("Blurred", image)

    canny = cv2.Canny(image, 30, 150)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)

def main():
    """
    load/display photos - ch2.
    """
    ap=argparse.ArgumentParser()                    # necessary inputs
    ap.add_argument("-i","--imgFolder",required=True,help="Path to the folder with a lot of images in it")
    ap.add_argument("-j","--bigImg",required=True,help="Path to image for big mosaic")
    args=vars(ap.parse_args())

    imgFolder=args["imgFolder"]
    bigImg=args["bigImg"]

    folderImgs=readImages(imgFolder)                # return image names from folder to make mosaic

    imgList=[]
    for img in folderImgs:
        dict={}
        imgName='images/'+img
        image = cv2.imread(imgName,cv2.IMREAD_UNCHANGED)
        imgColor=findDominantColor(image)
        dict["imgName"]=imgName
        dict["imgColor"]=imgColor
        imgList.append(dict)
    print('done processing folder imgs')
    displayDominant(imgList[0])
    displayRectangles(bigImg)
    #print('imgList: ',imgList)
    mosaic=replace(bigImg,imgList)
    cv2.imshow("Step 3: Match the Colors of the Rectangles to the Images",mosaic)
    cv2.waitKey(0)
    detectEdges(mosaic)

main()
