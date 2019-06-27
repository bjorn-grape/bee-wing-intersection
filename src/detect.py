#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import numpy as np
import multiprocessing
from skimage.io import imsave
from skimage.morphology import skeletonize
import argparse
from pathlib import Path


IMGS = None
CSV_FOLDER = None


def main(args):
    global IMGS, CSV_FOLDER
    # Handle images folder
    path_images = Path(args.images)
    if path_images.is_file() and path_images.suffix == '.jpg':
        IMGS = [path_images]
    elif path_images.is_dir():
        IMGS = [i for i in path_images.iterdir() if i.is_file() and i.suffix == '.jpg']
    else:
        print("Data directory does not exist.", file=sys.stderr)
        sys.exit(1)

    # Handle CSV folder
    CSV_FOLDER = Path(args.output)
    if not CSV_FOLDER.is_dir():
        os.makedirs(str(CSV_FOLDER))

    # Run veins detection
    print("Beginning CSV generation...")
    generateAllCsv(IMGS, 9)


def pathToNpArray(path):
    return np.array(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))


def kernell(i):
    return np.ones((i, i), np.uint8)


def differenceOfGaussian(img, kernSize1, kernSize2):
    g1 = cv2.blur(img, (kernSize1, kernSize1))
    g2 = cv2.blur(img, (kernSize2, kernSize2))
    return g1 - g2


def detectEdges(im):

    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(im)
    im = l

    im = differenceOfGaussian(im, 25, 30)
    im = cv2.bilateralFilter(im, 10, 50, 75)

    im = cv2.erode(im, kernell(3), iterations=1)
    im = cv2.dilate(im, kernell(2), iterations=1)

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


# Show only kernel 5x5 where number of pixel != 0 is greater or equal to 8
def find_points(im):
    img = (im > 0) * 1
    res = np.full(img.shape,0)
    height, width = img.shape
    for i in range(2, height-2):
        for j in range(2, width-2):
            subimg = img[i-2:i + 3, j-2:j + 3]
            ss = np.sum(subimg)
            res[i,j] = 255 if ss >= 8 else 0
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
    dotsImg = find_points(skell)
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
    image.save(outputPath)

def worker(folder, index, imgs):
    if index >= len(imgs):
        return
    path = imgs[index]
    img = pathToNpArray(path)
    centroids = imgToCentroidList(img, verbose=False)
    # centroids = np.flip(centroids, axis=1)
    csvPath = folder.joinpath(path.stem + '.csv')
    np.savetxt(str(csvPath), centroids, delimiter=",")
    print(index+1, "/", len(imgs), ": ", csvPath, " written.")


def generateAllCsv(imgs, processNumber):
    currIndex = 0
    while currIndex < len(imgs):
        processList = []
        for i in range(processNumber):
            p = multiprocessing.Process(target=worker, args=(CSV_FOLDER, currIndex + i, imgs))
            processList.append(p)
            p.start()

        for process in processList:
            process.join()
        currIndex += processNumber

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help='directory or a single image')
    parser.add_argument('--output', '-o', type=str, help='Output directory',
                        default='output')
    args = parser.parse_args()

    main(args)
