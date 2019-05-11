#! /bin/python3

import ImageLoader as IL
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_median_kernel(array2d, kernelsize):
    resarray = np.full(array2d.shape, -1)
    height = len(array2d)
    width = len(array2d[0])
    kern_div_2 = kernelsize / 2.0
    for i in range(height):
        for j in range(width):
            mini = int(max(i - kern_div_2, 0))
            maxi = int(min(i + kern_div_2, height))
            minj = int(max(j - kern_div_2, 0))
            maxj = int(min(j + kern_div_2, width))
            subarr = np.array(array2d[mini:maxi, minj:maxj]).flatten()
            subarr = np.sort(subarr)
            mid = int(len(subarr) / 2.0)
            median = subarr[mid]
            resarray[i,j] = median
    return resarray

def detectEdges1(image):
    imgBlurredL = cv2.blur(image, (9, 9))
    imgLab = cv2.cvtColor(imgBlurredL, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(imgLab)
    ret, thresh = cv2.threshold(l,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2 ,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    #markers = markers+1
    # Now, mark the region of unknown with zero
    #markers[unknown==255] = 0
    return unknown

def detectEdges2(image):
    #imgBlurredL = cv2.blur(image, (9, 9))

    im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(im)
    im = cv2.bilateralFilter(l, 13, 50, 75)
    kernel = np.ones((2,2),np.uint8)
    kernel2 = np.ones((7,7),np.uint8)
    im = cv2.erode(im,kernel,iterations = 1)
    im = cv2.blur(im, (7,7))
    im = cv2.medianBlur(im, 5)


    ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    

    im = thresh
    im = cv2.erode(im,kernel,iterations = 2)
    #im = cv2.bilateralFilter(im, 1, 1, 1)

    #im = cv2.bilateralFilter(im, (4, 4))
    im = cv2.dilate(im,kernel2,iterations = 1)
    im = cv2.erode(im,kernel,iterations = 2)
    #im = cv2.medianBlur(im, 5)


    #im = cv2.erode(im,kernel,iterations = 1)
    
    
    
    
    #ret, thresh = cv2.threshold(l,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #im = thresh
    #im  = apply_median_kernel(im, 2)

    return im


def detectEdges3(image):
    #imgBlurredL = cv2.blur(image, (9, 9))
    imgLab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(imgLab)
    ret, thresh = cv2.threshold(l,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.dilate(thresh,kernel,iterations = 1)
    dilate = cv2.erode(erosion,kernel,iterations = 1)

    return dilate

    #imgBlurredL = cv2.blur(l, (11, 11))
    #kernel = np.array([[1, 1, 1],
    #                  [1, -6, 1],
   #                   [1, 1, 1]])
   # dst = cv2.filter2D(imgBlurredL, -1, kernel)
   # ret3,th3 = cv2.threshold(dst,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
    #return th3
     
    
def allPlot(): # try me
    storage = IL.ImgStorage("../datas/train/")
    imgNB = storage.size() // 3
    fig, axes = plt.subplots(imgNB, 2, figsize=(8, 30))
    for i in range(imgNB):
        img = storage.getImgByIndex(i)
        res = detectEdges2(img)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(res)
    plt.show()

#allPlot()
    
def applyForOne():
    storage = IL.ImgStorage("../datas/train/")
    img = storage.getImgByIndex(8);
    plop = detectEdges2(img)
    plt.imshow(plop)
    
#applyForOne()
allPlot()