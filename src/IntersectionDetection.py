#!/usr/bin/env python3

from skimage.morphology import skeletonize

from EdgeDetection import  kernell, cleanWingImg

import ImageLoader
import numpy as np
import cv2
from os import listdir

def keepGT4(im):
    img = (im > 0) * 1
    res = np.full(img.shape,0)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            minx = max(0, i - 1)
            maxx = min(height - 1, i + 1)
            miny = max(0, j - 1)
            maxy = min(width - 1, j + 1)
            subimg = img[minx:maxx + 1,miny:maxy + 1]
            ss = np.sum(subimg)
            res[i,j] = 255 if ss >= 4 else 0
    return res.astype(np.uint8)

def imageDotsToCentroidList(img):
    im = cv2.dilate(img,kernell(5),iterations = 5)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    
    return centroids
    
def rawImgToCSV(imgPath, csvPath):
    img = ImageLoader.openPathToNpArray(imgPath)
    print("Cleaning image...")
    cleanedImg = cleanWingImg(img)
    print("computing skeleton...")
    cleanedImg = cleanedImg / 255
    skell = skeletonize(cleanedImg)
    print("keeping centroids...")
    dotsImg = keepGT4(skell)
    print("Computing centroids...")
    centroidList = imageDotsToCentroidList(dotsImg)
    print("Writing to file...")
    centroidList = np.flip(centroidList,axis= 1)
    np.savetxt(csvPath, centroidList, delimiter=",")
    print("Done.")
    

def computeAllFolder(folder):
    if(folder[-1] != "/"):
        folder += "/"
    listi = listdir(folder)
    listi = [elm for elm in listi if ".jpg" in elm or ".png" in elm]
    for index in range(len(listi)):
        print("Computing image ",index +1, "/", len(listi), "...")
        print(listi[index])
        rawImgToCSV(folder + listi[index], folder + listi[index][:-4] +".csv")
    
