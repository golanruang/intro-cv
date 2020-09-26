import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io
import math
# python final.py --imgFolder images --bigImg targetImage.jpeg --grid 10
def findMatch(img,imgList):
    minDist=-1
    index=0
    minIndex=-1
    for image in imgList:
        color=findDominantColor(img)
        #print(color[0])
        db=(color[0]-image["imgColor"][0])*(color[0]-image["imgColor"][0])
        dg=(color[1]-image["imgColor"][1])*(color[1]-image["imgColor"][1])
        dr=(color[2]-image["imgColor"][2])*(color[2]-image["imgColor"][2])
        dist=int(math.sqrt((db+dg+dr)))
        if dist>minDist:
            minDist=dist
            minIndex=index
        index+=1

    return minIndex

def replace(bigImg,imgList,grid):
    targetImage=cv2.imread(bigImg)
    width,height=targetImage.shape[0],targetImage.shape[1]
    unitW=int(int(width)/50)
    unitH=int(int(height)/50)
    imgs=[]
    for h in range(unitH):
        for w in range(unitW):
            try:
                topLeft=[w*unitW,h*unitH]
                bottomRight=[(w+1)*unitW,(h+1)*unitH]
                #print("topLeft: ",topLeft)
                #print("bottomRight: ",bottomRight)
                cv2.rectangle(targetImage,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,0,255),5)
                rect=targetImage[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
                img_to_replace_with=imgList[findMatch(rect,imgList)]["imgName"]
                finalImg=cv2.imread(img_to_replace_with)

                resized=cv2.resize(finalImg,(unitW,unitH),interpolation=cv2.INTER_AREA)
                targetImage[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]=resized
            except:
                hi=0
    resizedMosaic=cv2.resize(targetImage,(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("mosaic",resizedMosaic)
    cv2.waitKey()


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

def displayDominant(img):
    image=cv2.imread(img["imgName"])
    h=image.shape[0]
    w=image.shape[1]

    dom_img=np.zeros((h,w,3),np.uint8)
    dom_img[:] = img["imgColor"]

    cv2.imshow("Dominant Color Example", np.hstack([image, dom_img]))
    cv2.waitKey(0)

def main():
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
    displayDominant(imgList[0])

    #print('imgList: ',imgList)
    replace(bigImg,imgList)

main()
