#! /bin/python3

import ImageLoader as IL
import cv2
import numpy as np


storage = IL.ImgStorage("../datas/train/")

img = storage.getImgByIndex(11)
imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l,a,b = cv2.split(imgLab)

imgBlurredL = cv2.blur(l, (11, 11))

thresh0 = cv2.adaptiveThreshold(imgBlurredL, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

IL.showNpArrayAsImage(thresh0)


IL.showNpArrayAsImage(imgBlurredL)



kernel = np.array([[1, 1, 1],
                   [1, -6, 1],
                   [1, 1, 1]])

dst = cv2.filter2D(imgBlurredL, -1, kernel)

IL.showNpArrayAsImage(dst)



ret3,th3 = cv2.threshold(dst,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)

IL.showNpArrayAsImage(th3)

def tryImage(index):
    img = storage.getImgByIndex(index)
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(imgLab)
    
    imgBlurredL = cv2.blur(l, (11, 11))
    
    #thresh0 = cv2.adaptiveThreshold(imgBlurredL, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    IL.showNpArrayAsImage(thresh0)
    
    IL.showNpArrayAsImage(imgBlurredL)
    
    kernel = np.array([[1, 1, 1],
                       [1, -6, 1],
                       [1, 1, 1]])
    dst = cv2.filter2D(imgBlurredL, -1, kernel)
    IL.showNpArrayAsImage(dst)
    
    ret3,th3 = cv2.threshold(dst,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
    IL.showNpArrayAsImage(th3)
    
