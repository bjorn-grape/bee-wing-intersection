#! /bin/python3

import ImageLoader as IL

import numpy as np
import matplotlib.pyplot as plt
import cv2

def kernell(i):
    return np.ones((i,i),np.uint8)

def differenceOfGaussian(img, kernSize1, kernSize2):
    g1 = cv2.blur(img, (kernSize1, kernSize1))
    g2 = cv2.blur(img, (kernSize2, kernSize2))
    return g1 - g2;
     
#
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

def cleanWingImg(img):
    img1 = detectEdges(img)
    _, img2 = cv2.threshold(img1,127, 255,cv2.THRESH_BINARY)
    _, labels = cv2.connectedComponents(img2)
    cppm = getComplement(labels)
    img3 = np.subtract(img2,cppm)
    img3 = img3.astype(np.uint8)
    img3 = cv2.dilate(img3, kernell(4), iterations = 3)
    img3 = cv2.erode(img3, kernell(4), iterations = 3)
    img3 = cv2.medianBlur(img3, 25)
    img3 = cv2.medianBlur(img3, 25)
    return img3

def allPlot(): # try me ~ 7 seconds per image
    storage = IL.ImgStorage("../datas/train/")
    imgNB = storage.size() // 3 
    fig, axes = plt.subplots(imgNB, 2, figsize=(20, 80))
    for i in range(imgNB):
        print("Computing image ", (i+1), "out of ",imgNB, "...")
        img = storage.getImgByIndex(i)
        res = cleanWingImg(img)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(res)
        print("Done.")
    plt.show()

def onePlot():
    storage = IL.ImgStorage("../datas/train/")
    img = storage.getImgByIndex(0);
    res = img
    res = cleanWingImg(res)
    plt.imshow(res)