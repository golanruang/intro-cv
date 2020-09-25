"""
I looked at https://www.geeksforgeeks.org/implementing-photomosaics/ to see the steps
to create a photomosaic (i didn't use their code becuase I had no idea what was going on in it)
"""
import os, random, argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from skimage import io

# python mosaic.py --imgFolder images --bigImg targetImage.jpeg --grid 10

def readImages(dir):
    images = []
    for f in os.listdir(dir):                          # all the image names
            images.append(f)
    return images

def splitImage(image,grid):
    print('splitting big image into small rectangular ones')
    width, height=image.shape[0],image.shape[1]
    w,h=int(int(width)/int(grid)),int(int(height)/int(grid))     # cutting the images into sections
    gridImgs=[]
    for i in range(w):
        for j in range(h):
            if (j+1)*h < height and (i+1)*w<width:
                # print("top left y",j*h)
                # print("bottom right y",(j+1)*h)
                # print("top left x",i*w)
                # print("bottom right x",(i+1)*w)
                crop_img = image[j*h:(j+1)*h, i*w:(i+1)*w]       # cropping images - chapter 4
                gridImgs.append(crop_img)                        # append the numpy arrays to a python list
    return gridImgs

# def createMosaic(imgList, image, grid):
#
#     width, height=image.shape[0],image.shape[1]
#     w,h=int(int(width)/int(grid)),int(int(height)/int(grid))
#     gridImgs=[]
#     for i in range(w):
#         for j in range(h):
#             if (j+1)*h < height and (i+1)*w<width:

def findBGR(img):
    """
    Finds the dominant color of a image (dominant != average)
    https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html
    """
    img=np.array(img)
    try:
        height=np.shape(img)[0]
        width=np.shape(img)[1]
    except:
        return -1

    # calculate the average color of each row of our image
    avg_color_per_row = np.average(img, axis=0)

    # calculate the averages of our rows
    avg_colors = np.average(avg_color_per_row, axis=0)

    return avg_colors
    # avg_color is a tuple in BGR order of the average colors
    # but as float values
    # print(f'avg_colors: {avg_colors}')

    # so, convert that array to integers
    # int_averages = np.array(avg_colors, dtype=np.uint8)
    # print(f'int_averages: {int_averages}')

    # create a new image of the same height/width as the original
    # average_image = np.zeros((height, width, 3), np.uint8)
    # and fill its pixels with our average color
    # average_image[:] = int_averages


    # finally, show it side-by-side with the original
    # cv2.imshow("Avg Color", np.hstack([img, average_image]))
    # cv2.waitKey(0)

# color modifications

def removeBad(imgList):
    for i in range(0,len(imgList)-1):
        if type(imgList[i]["imgColors"])==int:
            imgList.pop(i)
    return imgList

def findMatch(imgList,rectColor):
    print('finding the best matched image')
    distances=[]
    for img in imgList:
        db=(rectColor[0]-img["imgColors"][0])*(rectColor[0]-img["imgColors"][0])
        dg=(rectColor[1]-img["imgColors"][1])*(rectColor[1]-img["imgColors"][1])
        dr=(rectColor[2]-img["imgColors"][2])*(rectColor[2]-img["imgColors"][2])
        dist=db+dg+dr
        distances.append(dist)
    min=distances[0]
    minIndex=-1
    index=0
    for val in distances:
        if val>min:
            min=val
            minIndex=index
        index+=1

    return minIndex
    # TODO: finish this function

def replace(targetImage,grid,imgList):
    width, height=targetImage.shape[0],targetImage.shape[1] #
    w,h=int(int(width)/int(grid)),int(int(height)/int(grid))
    gridImgs=[]
    for i in range(w):
        for j in range(h):
            tl=(int(i*w),int(j*h))
            br=(int((i+1)*w),int((j+1)*h))
            tr=(int((i+1)*w),int(j*h))
            bl=(int(i*w),int((j+1)*h))
            rect=targetImage[tl[1]:br[1], tl[0]:br[0]]
            BGR=findBGR(rect)
            replaceImg=findMatch(imgList,BGR)
            dim=((i+w)*w)
            resized=cv2.resize(replaceImg,(int(w),int(h)))
            print("resized: ",resized)
            #resized.append(3)
            targetImage[tl[1]:br[1], tl[0]:br[0]]=resized
    cv2.imshow(targetImage)
    cv2.waitKey()

    # for rect in targetSplit:
    #     rectColor=findBGR(rect)
    #     for img in imgList:
    #         dist=(rectColor[0]-img["imgColors"][0])*(rectColor[0]-img["imgColors"][0])
    #         +(rectColor[1]-img["imgColors"][1])*(rectColor[1]-img["imgColors"][1])
    #         +(rectColor[2]-img["imgColors"][2])*(rectColor[2]-img["imgColors"][2])
            #if dist<min_dist:

def main():
    ap=argparse.ArgumentParser()                    # necessary inputs
    ap.add_argument("-i","--imgFolder",required=True,help="Path to the folder with a lot of images in it")
    ap.add_argument("-j","--bigImg",required=True,help="Path to image for big mosaic")
    ap.add_argument("-k","--grid",required=True,help="Size of mosaic")
    args=vars(ap.parse_args())

    imgFolder=args["imgFolder"]
    bigImg=args["bigImg"]
    grid=args["grid"]
    folderImgs=readImages(imgFolder)                # images from folder to make mosaic

    index=0                                         # give a name to each np array
    imgList=[]                                      # list of dicts with image info in each dict
    smallerImgs=[]
    for img in folderImgs:
        image=cv2.imread('images/' + img)           # read each img
        imgDict={}
        avgColor=findBGR(image)                     # find the dominant color
        imgDict["name"]=str(index)
        imgDict["imgColors"]=avgColor
        imgDict["npArray"]=np.array(image)
        imgList.append(imgDict)                     # make a dict that gives the name, dominant color, and array of each img
    #print(imgList)
    bigImg=cv2.imread(bigImg)
    # targetSplit=splitImage(bigImg,grid)
    # min_dist=float("inf")
    print('before: ',imgList)
    imgList=removeBad(imgList)
    print('new img list:',imgList)
    mosaic=replace(bigImg,grid,imgList)
    # for rect in targetImage:
    #     rectColor=findBGR(rect)
    #     for img in imgList:
    #         dist=(rectColor[0]-img["imgColors"][0])*(rectColor[0]-img["imgColors"][0])
    #         +(rectColor[1]-img["imgColors"][1])*(rectColor[1]-img["imgColors"][1])
    #         +(rectColor[2]-img["imgColors"][2])*(rectColor[2]-img["imgColors"][2])
            #if dist<min_dist:
                #



        # actualImgs.append(image)
    #
    # smallerImgs=[]
    #
    # for img in actualImgs:
    #     split=splitImage(img,grid)
    #     smallerImgs.append(split)
    #
    # #dict={"imgName":
    # #      "imgColors: "
    # #      "npArray: "}
    #
    # for img in smallerImgs:
    #     print('finding rbg')
    #     avgColor=findRGB(img)
    #     print(avgColor)


main()
