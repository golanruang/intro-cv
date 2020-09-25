import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io
# python final.py --imgFolder images --bigImg targetImage.jpeg --grid 10
def findMatch(img,imgList):
    minDist=-1
    index=0
    minIndex=-1
    for image in imgList:
        color=findDominantColor(img)
        print(color[0])
        db=(color[0]-image["imgColor"][0])*(color[0]-image["imgColor"][0])
        dg=(color[1]-image["imgColor"][1])*(color[1]-image["imgColor"][1])
        dr=(color[2]-image["imgColor"][2])*(color[2]-image["imgColor"][2])
        dist=int(db+dg+dr)
        if dist>minDist:
            minDist=dist
            minIndex=index
        index+=1

    return minIndex

def replace(bigImg,imgList,grid):
    targetImage=cv2.imread(bigImg)
    width,height=targetImage.shape[0],targetImage.shape[1]
    unitW=int(int(width)/40)
    unitH=int(int(height)/40)
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
                print(':(')
    resizedMosaic=cv2.resize(targetImage,(1060,707),interpolation=cv2.INTER_AREA)
    cv2.imshow("mosaic",resizedMosaic)
    cv2.waitKey()

    # cv2.imshow("gridLines",targetImage)
    # cv2.waitKey()


def readImages(dir):
    images = []
    for f in os.listdir(dir):                          # all the image names
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
            images.append(f)
    return images

def findDominantColor(img):

    data = np.reshape(img, (-1,3))
    #print(data.shape)
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

    #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    #print(centers[0].astype(np.int32)[0])
    return centers[0].astype(np.int32)

def main():
    ap=argparse.ArgumentParser()                    # necessary inputs
    ap.add_argument("-i","--imgFolder",required=True,help="Path to the folder with a lot of images in it")
    ap.add_argument("-j","--bigImg",required=True,help="Path to image for big mosaic")
    ap.add_argument("-k","--grid",required=True,help="Size of mosaic")
    args=vars(ap.parse_args())

    imgFolder=args["imgFolder"]
    bigImg=args["bigImg"]
    yes=cv2.imread(bigImg)
    h,w=yes.shape[0],yes.shape[1]

    grid=args["grid"]
    folderImgs=readImages(imgFolder)                # return image names from folder to make mosaic

    imgList=[]
    #print('bye')
    for img in folderImgs:
        dict={}
        imgName='images/'+img
        image = cv2.imread(imgName,cv2.IMREAD_UNCHANGED)
        imgColor=findDominantColor(image)
        dict["imgName"]=imgName
        dict["imgColor"]=imgColor
        imgList.append(dict)

    mosaic=replace(bigImg,imgList,grid)

main()
