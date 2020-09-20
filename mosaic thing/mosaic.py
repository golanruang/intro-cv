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

# python mosaic.py --imgFolder images --bigImg targetImage --grid 10

def readImages(dir):
    images = []
    for f in os.listdir(dir):                          # for every file in the path given
            images.append(f)
    return images

def splitImage(image,grid):
    print('splitting big image into small rectangular ones')
    width, height=image.shape[0],image.shape[1]
    print(width)
    print(height)
    w,h=int(int(width)/int(grid)),int(int(height)/int(grid))
    gridImgs=[]
    for i in range(w):
        for j in range(h):
            if (j+1)*h < height and (i+1)*w<width:
                # print("top left y",j*h)
                # print("bottom right y",(j+1)*h)
                # print("top left x",i*w)
                # print("bottom right x",(i+1)*w)
                crop_img = image[j*h:(j+1)*h, i*w:(i+1)*w]       # cropping images - chapter 4
                gridImgs.append(crop_img)
    return gridImgs

def findRGB(img):
    """
    Finds the dominant color of a image (dominant != average)
    https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    """
    # img=io.imread(image)[:,:,:-1]
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    print(avg_color_per_row)


    #
    # im = np.array(image)
    # print(im)
    # print(im.shape)
    # width=im.shape[1]
    # height=im.shape[0]
    # channels=im.shape[2]
    # numPix=width*height
    # for w in width:
    #     for h in height:
    #
    # avg = (tuple(np.average(im.reshape(b * g, r), axis=0)))
    # return avg

def findMatch():
    print('finding the best matched image')
    pass

def combine():
    print('combining the images from folder')
    pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--imgFolder",required=True,help="Path to the folder with a lot of images in it")
    ap.add_argument("-j","--bigImg",required=True,help="Path to image for big mosaic")
    ap.add_argument("-k","--grid",required=True,help="Size of mosaic")
    args=vars(ap.parse_args())
    imgFolder=args["imgFolder"]
    bigImg=args["bigImg"]
    grid=args["grid"]
    folderImgs=readImages(imgFolder)
    print(folderImgs)
    actualImgs=[]
    for img in folderImgs:
        image=cv2.imread('images/' + img)
        actualImgs.append(image)

    smallerImgs=[]

    for img in actualImgs:
        split=splitImage(img,grid)
        print(split)
        smallerImgs.append(split)

    for img in smallerImgs:
        img=np.array(img)
        avgColor=findRGB(img)


main()
