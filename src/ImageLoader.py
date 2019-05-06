#! /bin/python

from PIL import Image
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

 

def openPathToNpArray(path):
    return np.array(Image.open(path))

def getAllImages(basePath):
    trainFiles = listdir(basePath)
    storage = {}
    for elm in trainFiles:
        storage[elm] = openPathToNpArray(basePath + elm)
    return storage

def showNpArrayAsImage(npArray):
    plt.imshow(npArray)

def displayAllImages(allImg): # dictionary of {"imgName" : imgAsNpArray,...}
    for name in allImg:
        showNpArrayAsImage(name, allImg[name])
   
class ImgStorage:
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
    
    def printImgWithIndex(self, index):
        showNpArrayAsImage(self.getImgByIndex(index))
        
    def printImgWithName(self, name):
        showNpArrayAsImage(self.getImgByName(name))

    


