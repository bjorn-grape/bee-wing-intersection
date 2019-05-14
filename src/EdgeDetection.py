#! /bin/python3

import ImageLoader as IL
import cv2
import numpy as np
import matplotlib.pyplot as plt

from queue import *



def kernell(i):
    return np.ones((i,i),np.uint8)

def differenceOfGaussian(img, kernSize1, kernSize2):
    g1 = cv2.blur(img, (kernSize1, kernSize1))
    g2 = cv2.blur(img, (kernSize2, kernSize2))
    return g1 - g2;

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
    im = cv2.bilateralFilter(l, 10, 50, 75)
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
    im = cv2.erode(im,kernel,iterations = 1)
    #im = cv2.medianBlur(im, 5)


    #im = cv2.erode(im,kernel,iterations = 1)
    
    
    
    
    #ret, thresh = cv2.threshold(l,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #im = thresh
    #im  = apply_median_kernel(im, 2)

    return im

def getColorForValue(val, maxval):
    hue = val * 255 // maxval
    saturation = 255
    value = 255
    return np.array([hue,saturation,value])
    

def giveValueBy8Connexity(img):
    height, width = img.shape
    res = np.full((height, width), 0)
    q = Queue()
    height, width = img.shape
    current_color = 1
    for i in range(height):
        for j in range(width):
            while not q.empty():
                coordx, coordy = q.get()
                minx = max(0,(coordx - 1))
                maxx = min(height,(coordx + 2))
                miny = max(0,(coordy - 1))
                maxy = min(width,(coordy + 2))
                for k in  range(minx,maxx):
                    for l in range(miny,maxy):
                        #print(str(k)+","+ str(l) + "| img[k,l] != 0 -> "+str(img[k,l] != 0) +" |  res[k,l] == 0 -> " +str( res[k,l] == 0)  )
                        if img[k,l] != 0 and res[k,l] == 0:
                            res[k,l] = img[coordx,coordy]
                            #print("In: putting " +str(k)+","+ str(l))
                            q.put((k,l))
            if img[i,j] != 0 and res[i,j] == 0:
                res[i,j] = current_color
                #print("color: " + str(current_color))
                current_color += 1
                
                q.put((i,j))      
                #print("Out: putting " +str(i)+","+ str(j))
            
    return res

def detectEdges3(image):
    
    im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(im)
    im = l
    im = cv2.bilateralFilter(im, 10, 50, 75)
    im = cv2.erode(im,kernell(2),iterations = 1)
    #im = cv2.blur(im, (7,7))
   
    ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    im = thresh
    im = cv2.erode(im,kernell(2),iterations = 2)

    im = cv2.dilate(im,kernell(2),iterations = 4)
    #im = cv2.erode(im,kernell(2),iterations = 1)

    return im
     
def detectEdges4(image):
    im = image
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    #im = cv2.fastNlMeansDenoisingColored(im,None,7,7,25,12)
    
    l,a,b = cv2.split(im)
    im = l
    im = differenceOfGaussian(im, 25, 30)
    im = cv2.bilateralFilter(im, 10, 50, 75)

    im = cv2.erode(im,kernell(3),iterations = 1)
   
  #  ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
   # im = thresh

    im = cv2.dilate(im,kernell(2),iterations = 1)
    im = cv2.medianBlur(im, 3)
    im = cv2.medianBlur(im, 3)
    im = cv2.medianBlur(im, 3)
    im = cv2.medianBlur(im, 3)
    im = cv2.dilate(im,kernell(3),iterations = 2)
    im = cv2.medianBlur(im, 3)
    im = cv2.medianBlur(im, 3)



   # im = cv2.erode(im,kernell(7),iterations = 2)

    return im
    
def allPlot(): # try me
    storage = IL.ImgStorage("../datas/train/")
    imgNB = storage.size() // 3
    fig, axes = plt.subplots(imgNB, 2, figsize=(20, 80))
    for i in range(imgNB):
        img = storage.getImgByIndex(i)
        res = detectEdges4(img)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(res)
    plt.show()

#allPlot()
    
def applyForOne():
    storage = IL.ImgStorage("../datas/train/")
    img = storage.getImgByIndex(0);
    res = detectEdges4(img)
    plt.imshow(res)
    
def testy():
    storage = IL.ImgStorage("../datas/train/")
    img = storage.getImgByIndex(0);
    res = img
    res = cleanWingImg(res)
    #res = detectEdges4(res)
    #res = res[72:83,73:85]

    #res = giveValueBy8Connexity(res)
    #return  res
    plt.imshow(res)
    
#applyForOne()
#allPlot()
#print(testy())

def colorize(img):
    maxi = np.max(img) + 1
    height, width = img.shape
    res = np.full((height, width, 3), 0)

    for i in range(height):
        for j in range(width):
            color = getColorForValue(img[i,j], maxi)
            for k in range(3):
                res[i,j,k] = color[k]
    #res = cv2.cvtColor(res , cv2.COLOR_HSV2RGB)
    return res

def histogram(img):
    maxi = int(np.max(img) + 1)
    img = img.flatten()
    res = np.full((maxi), 0)
    for i in range(len(img)):
        res[int(img[i])] += 1
    return res
        
def getComplement(img):
    height,width = img.shape
    res = np.full((height, width), 0)
    res = np.array(res, dtype=np.int32)
    histi = histogram(img)
    mean = np.mean(histi) * 2
    for i in range(height):
        for j in range(width):
            valuee = img[i,j]
            if  histi[valuee] < mean:
                res[i,j]  = 255
            else:
                res[i,j]  = 0
    return res


def give_components(img):
    ret, labels = cv2.connectedComponents(img)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def cleanWingImg(img):
    img1 = detectEdges4(img)
    q, img2 = cv2.threshold(img1,127, 255,cv2.THRESH_BINARY)
    _, labels = cv2.connectedComponents(img2)
    cppm = getComplement(labels)
    img3 = np.subtract(img2,cppm)
    return img3