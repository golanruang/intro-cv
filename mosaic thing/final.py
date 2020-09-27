import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io
import math
#from __future__ import print_function
from matplotlib import pyplot as plt
# python final.py --imgFolder images --bigImg targetImage.png
def findMatch(img,imgList):
    minDist=100000
    index=0
    minIndex=-1
    color=findDominantColor(img)
    for image in imgList:
        print("image color: ",image["imgColor"])
        db=(color[0]-image["imgColor"][0])*(color[0]-image["imgColor"][0])
        dg=(color[1]-image["imgColor"][1])*(color[1]-image["imgColor"][1])
        dr=(color[2]-image["imgColor"][2])*(color[2]-image["imgColor"][2])
        dist=int(math.sqrt((db+dg+dr)))
        if dist<minDist:
            minDist=dist
            minIndex=index
        index+=1
    print("img chosen: ", imgList[minIndex]["imgName"])
    return minIndex

def displayRectangles(img,boxW,boxH):
    """
    dRaWs rEcTaNgLeS
    ch5.
    """
    image=cv2.imread(img)
    width,height=image.shape[1],image.shape[0]
    print("width: ",width)
    print("height: ",height)
    print("boxH: ",boxH)
    print("boxW: ",boxW)
    unitW=int(width/boxW)
    unitH=int(height/boxH)
    print("unitH: ",unitH)
    print("unitW: ",unitW)
    imgs=[]
    for h in range(unitH):
        for w in range(unitW):
            topLeft=[w*boxW,h*boxH]
            bottomRight=[(w+1)*boxW,(h+1)*boxH]
            cv2.rectangle(image,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,0,255),5)
    resizedImg=cv2.resize(image,(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("Bounding Rectangles",resizedImg)
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


def replace(bigImg,imgList,boxH,boxW):
    """
    img cropping - ch6
    """
    targetImage=cv2.imread(bigImg)
    width,height=targetImage.shape[1],targetImage.shape[0]
    unitW=int(int(width)/boxH)
    unitH=int(int(height)/boxW)
    imgs=[]
    for h in range(boxH):
        for w in range(boxW):
            try:
                topLeft=[w*unitW,h*unitH]
                bottomRight=[(w+1)*unitW,(h+1)*unitH]
                #print("topLeft: ",topLeft)
                #print("bottomRight: ",bottomRight)
                #cv2.rectangle(targetImage,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,0,255),5)
                rect=targetImage[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
                img_to_replace_with=imgList[findMatch(rect,imgList)]["imgName"]
                finalImg=cv2.imread(img_to_replace_with)

                resized=cv2.resize(finalImg,(unitW,unitH),interpolation=cv2.INTER_AREA)
                targetImage[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]=resized
            except:
                hi=0
    resizedMosaic=cv2.resize(targetImage,(1060,707),interpolation=cv2.INTER_AREA)
    return resizedMosaic
    print('finished making mosaic')

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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

    #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))

    return centers[0].astype(np.int32)

def findRectDim(image):
    """
    Find how many rectangles should be for every image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 9, 75)
    cv2.imshow("Image", image)

    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Edges", edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured = image.copy()
    cv2.drawContours(contoured, cnts, -1, (0, 255, 0), 2)
    numBoxes=0
    totalWidths=0
    totalHeights=0
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        totalWidths+=w
        totalHeights+=h
        numBoxes+=1
    height=(totalHeights/numBoxes) * 13/10
    width=(totalWidths/numBoxes) * 13/10
    #print("height",height)
    #print("width",width)
    #return height,width
    return 115, 115

def displayDominant(img):
    image=cv2.imread(img["imgName"])
    h=image.shape[0]
    w=image.shape[1]

    dom_img=np.zeros((h,w,3),np.uint8)
    dom_img[:] = img["imgColor"]

    resizedImg=cv2.resize(np.hstack([image, dom_img]),(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("Dominant Color Example of One Img", resizedImg)
    cv2.waitKey(0)

def main():
    ap=argparse.ArgumentParser()                    # necessary inputs
    ap.add_argument("-i","--imgFolder",required=True,help="Path to the folder with a lot of images in it")
    ap.add_argument("-j","--bigImg",required=True,help="Path to image for big mosaic")
    args=vars(ap.parse_args())

    imgFolder=args["imgFolder"]
    bigImg=args["bigImg"]

    folderImgs=readImages(imgFolder)                # return image names from folder to make mosaic

    boxH, boxW=findRectDim(cv2.imread(bigImg))

    boxH=int(boxH)
    boxW=int(boxW)
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
    displayRectangles(bigImg,boxW,boxH)
    #print('imgList: ',imgList)
    mosaic=replace(bigImg,imgList,boxH,boxW)
    cv2.imshow("mosaic",resizedMosaic)
    cv2.waitKey(0)

main()
