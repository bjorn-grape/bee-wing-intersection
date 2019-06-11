#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import argparse


IMG_FOLDER = None
CSV_FOLDER = "output"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    args = parser.parse_args()
    
    IMG_FOLDER = args.image_folder
    if IMG_FOLDER[-1] != '/':
        IMG_FOLDER += '/'



def openPathToNpArray(path):
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

def getAllImages(basePath):
    trainFiles = [image for image in listdir(basePath) if image.endswith('.jpg')]
    storage = {}
    for elm in trainFiles:
        storage[elm] = openPathToNpArray(basePath + elm)
    return storage

def displayAllImages(allImg): # dictionary of {"imgName" : imgAsNpArray,...}
    for name in allImg:
        showNpArrayAsImage(name, allImg[name])
   
class ImgStorage: # i.e.: storage =  ImgStorage("../datas/train/")
    def __init__(self, basePath):
        self.localMap = getAllImages(basePath)
        self.localList = [(name, self.localMap[name]) for name in self.localMap]
        
    def getImgByIndex(self, index):
        key,value = self.localList[index]
        return value
    
    def getImgNameByIndex(self, index):
        key,value = self.localList[index]
        return key
    
    def getImgByName(self, name):
        return self.localMap[name]
    
    def size(self):
        return len(self.localList)
    
    def showIndex(self, index):
        plt.imshow(self.getImgByIndex(index))        
        
    def showName(self, name):
        plt.imshow(self.getImgByName(name))
        
    def getCSVNameForIndex(self, index):
        return self.getImgNameByIndex(index)[:-4] + ".csv"
    
def kernell(i):
    return np.ones((i,i),np.uint8)

def differenceOfGaussian(img, kernSize1, kernSize2):
    g1 = cv2.blur(img, (kernSize1, kernSize1))
    g2 = cv2.blur(img, (kernSize2, kernSize2))
    return g1 - g2;

def detectEdges(im):
    
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(im)
    im = l
    
    im = differenceOfGaussian(im, 25, 30)
    im = cv2.bilateralFilter(im, 10, 50, 75)

    im = cv2.erode(im,kernell(3),iterations = 1)
    im = cv2.dilate(im,kernell(2),iterations = 1)
    
    for i in range(4):
        im = cv2.medianBlur(im, 3)
    
    im = cv2.dilate(im,kernell(3),iterations = 2)
    
    im = cv2.medianBlur(im, 3)
    im = cv2.medianBlur(im, 3)

    return im

def histogram(img):
    maxi = int(np.max(img) + 1)
    img = img.flatten()
    res = np.full((maxi), 0)
    for i in range(len(img)):
        res[int(img[i])] += 1
    return res

# create image to substract (it contains only noise)
def getComplement(img):
    height,width = img.shape
    res = np.full((height, width), 0)
    res = np.array(res, dtype=np.int32)
    histi = histogram(img)
    sortedhisto = np.sort(histi)
    # keep only the 3 biggest parts
    mean = sortedhisto[-3]
    for i in range(height):
        for j in range(width):
            valuee = img[i,j]
            # if pixel is part not part of the 3 biggest parts keep it
            if  histi[valuee] < mean:
                res[i,j]  = 255
            else:
                res[i,j]  = 0
    return res

def postClean(img):
    img = img.astype(np.uint8)
    img = cv2.dilate(img, kernell(4), iterations = 3)
    img = cv2.erode(img, kernell(4), iterations = 3)
    img = cv2.medianBlur(img, 25)
    img = cv2.medianBlur(img, 25)
    return img

def cleanWingImg(img):
    img1 = detectEdges(img)
    _, img2 = cv2.threshold(img1,127, 255,cv2.THRESH_BINARY)
    _, labels = cv2.connectedComponents(img2)
    cppm = getComplement(labels)
    img3 = np.subtract(img2,cppm)
    cleaned = postClean(img3)
    return cleaned

from skimage.morphology import skeletonize

# Show only kernel 3x3 where number of pixel != 0 is greater or equal to 4
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

# keep centroids of connected components
def imageDotsToCentroidList(img):
    im = cv2.dilate(img,kernell(5),iterations = 5)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    return centroids

def imgToCentroidList(img, verbose=True):
    if(verbose):
        print("Cleaning image...")
    cleanedImg = cleanWingImg(img)
    if(verbose):
        print("computing skeleton...")
    cleanedImg = cleanedImg / 255
    skell = skeletonize(cleanedImg)
    if(verbose):
        print("keeping centroids...")
    dotsImg = keepGT4(skell)
    if(verbose):
        print("Computing centroids...")
    centroidList = imageDotsToCentroidList(dotsImg)
    if(verbose):
        print("Writing to file...")
    centroidList = np.flip(centroidList,axis= 1)
    if(verbose):
        print("Done.")
    return centroidList

import csv
import PIL
from PIL import Image, ImageDraw
from numpy import loadtxt, array

def draw_intersections(image, intersections, width):
    image = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for x, y in intersections:
        draw.ellipse((y - width, x - width, y + width, x + width), fill=(255, 0, 0))
    return image

def generateResultImage(imagePath, csvPath, outputPath):
    image = Image.open(imagePath)
    coords = loadtxt(csvPath, delimiter=',')
    if len(coords.shape) != 2:
        coords = array([coords])
    draw_intersections(image, coords)
    image.save(output_path)

def worker(folder,index, imgStorage, size):
    if index >= size:
        return
    im = imgStorage.getImgByIndex(index)
    centroids =  imgToCentroidList(im, verbose=False)
    #centroids = np.flip(centroids, axis=1)
    csvPath = imgStorage.getCSVNameForIndex(index)
    #print(csvPath)
    np.savetxt(folder + csvPath, centroids, delimiter=",")
    print(index+1, "/", imgStorage.size(), ": ",csvPath ," written.")
    
import multiprocessing

def generateAllCsv(imgStorage, processNumber):
    folder = CSV_FOLDER
    if folder[-1] != "/":
            folder = folder + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    currIndex = 0
    while currIndex < imgStorage.size():
        processList = []
        for i in range(processNumber):
            p = multiprocessing.Process(target=worker, args=(folder,currIndex + i, imgStorage, imgStorage.size()))
            processList.append(p)
            p.start()

        for process in processList:
            process.join()
        currIndex += processNumber
    
    print("Done.")

allImg = ImgStorage(IMG_FOLDER)

print("Beginning CSV generation...")
generateAllCsv(allImg, 9)
