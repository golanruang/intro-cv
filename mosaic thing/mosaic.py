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
from sklearn.cluster import KMeans

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
    https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html
    """
    height, width, _ = np.shape(img)

    # calculate the average color of each row of our image
    avg_color_per_row = np.average(img, axis=0)

    # calculate the averages of our rows
    avg_colors = np.average(avg_color_per_row, axis=0)

    # avg_color is a tuple in BGR order of the average colors
    # but as float values
    print(f'avg_colors: {avg_colors}')

    # so, convert that array to integers
    int_averages = np.array(avg_colors, dtype=np.uint8)
    print(f'int_averages: {int_averages}')

    # create a new image of the same height/width as the original
    average_image = np.zeros((height, width, 3), np.uint8)
    # and fill its pixels with our average color
    average_image[:] = int_averages

    # finally, show it side-by-side with the original
    # cv2.imshow("Avg Color", np.hstack([img, average_image]))
    # cv2.waitKey(0)

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
