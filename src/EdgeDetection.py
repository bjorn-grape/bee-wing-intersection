#! /bin/python3

import ImageLoader as IL
import cv2
import numpy as np
import matplotlib.pyplot as plt



def detectEdges(image):
    imgLab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(imgLab)
    imgBlurredL = cv2.blur(l, (11, 11))
    kernel = np.array([[1, 1, 1],
                      [1, -6, 1],
                      [1, 1, 1]])
    dst = cv2.filter2D(imgBlurredL, -1, kernel)
    ret3,th3 = cv2.threshold(dst,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
    return th3
     
    
def allPlot(): # try me
    storage = IL.ImgStorage("../datas/train/")
    imgNB = storage.size() / 3
    fig, axes = plt.subplots(imgNB, 2, figsize=(8, 30))
    for i in range(imgNB):
        img = storage.getImgByIndex(i)
        res = detectEdges(img)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(res)
    plt.show()

