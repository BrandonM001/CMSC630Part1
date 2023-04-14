# Brandon Mohan
# CMSC 630 Project Part 1
#Sources used:
#https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
#https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
#https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
#https://www.geeksforgeeks.org/edge-detection-using-prewitt-scharr-and-sobel-operator/
#https://en.wikipedia.org/wiki/Prewitt_operator
#https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/

from part1 import * #need monoColor from last part
import math
import sys #used to get inputFile
from PIL import Image #used to open Images
import matplotlib.pyplot as plot
import numpy as np #used to convert images to numbers for processing
import shlex #used to help parse input
import random
import time #used to get time metrics


if len(sys.argv) < 2: #make sure we have an input file
    print(sys.argv)
    print("")
    sys.exit(1)

def main():  # main function
    startTime = time.time()#start time  to calculate image processing duraction
    endTime = time.time() #start time for calculating overall program duration
    inputFile = sys.argv[1]
    numImages = 0
    names = [] #keep track of filenames
    timeCount = [] #tracks amount of time spent on each function
    imageARY = []
    with open(inputFile, 'r') as input:
        for line in input: #get grayscale of each image first
            startTime = time.time()
            imageTime = []
            numImages += 1
            parts = shlex.split(line) #break each line of input into array
            filename = parts[0]# name/path of image file
            names.append(filename)
            img = openImg(filename) #function from part1
            print("Running: " + filename)
            timings = time.time() #take starting time
            monoImg = monoColor(img, parts[1]) #converts color images to selected single color
            #monoColor from part1
            imageTime.append(time.time() - timings) #measures how long the last function ran
            timings = time.time() #reset timer
            imageARY.append(monoImg)
            #imageTime.append(imageTime)
            sobel = sobelFilter(monoImg) #apply sobel operator
            imageTime.append(time.time() - timings)
            timings = time.time()
            dilate = dilationErosion(monoImg, monoImg, True) #apply dilation
            imageTime.append(time.time() - timings)
            timings = time.time()
            erosion = dilationErosion(monoImg, monoImg, False) #apply erosion
            imageTime.append(time.time() - timings)
            timings = time.time()
            #histoThres = histogram_thresholding(img)
            cluster = clustering(monoImg, 5, 3)
            imageTime.append(time.time() - timings)
            timeCount.append(imageTime)
            Image.fromarray(monoImg, 'RGB').show() #print original image
            Image.fromarray(sobel, 'RGB').show() #print edges
            Image.fromarray(dilate, 'RGB').show() #print dilated image
            Image.fromarray(erosion, 'RGB').show() #print eroded image
            #plot.bar(range(256), histoThres)
            Image.fromarray(cluster, 'RGB').show() #print foreground of image
        for pict in range(0, numImages):#display how long each function took for each image
            print(names[pict])
            print("Grayscale took " + str(timeCount[pict][0]) + " seconds")
            print("Sobel took " + str(timeCount[pict][1]) + " seconds")
            print("Dilation took " + str(timeCount[pict][2]) + " seconds")
            print("Erosion took " + str(timeCount[pict][3]) + " seconds")
            print("Clustering took " + str(timeCount[pict][4]) + " seconds")
            img = imageARY[pict]
        return 0

def sobelFilter(img):
    xAxis = img.shape[0]#get general size info
    yAxis = img.shape[1]
    #define sobel kernels
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernelM = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    kernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradX = np.array(img) #make kernels for each dimension
    gradY = np.array(img)
    kerX = kernelX.shape[0]#get kernel size
    kerY = kernelY.shape[1]
    copy = np.array(img)
    # Apply the kernels to the image
    for i in range(1, xAxis-1):
        for j in range(1, yAxis-1): #loop through each pixel in image
            region = np.array(img)
            xImg = np.sum(np.sum(kernelX * img[i-1:i+2, j-1:j+2]))
            yImg = np.sum(np.sum(kernelY * img[i-1:i+2, j-1:j+2]))
            #"img[i-1:i+2, j-1:j+2]" gets 3x3 blocks of pixels around the pixel at i, j
            #part1 would've been simplier if I knew i could do this in python
            copy[i, j] = (np.sqrt(np.square(xImg) + np.square(yImg)))
    return copy

def dilationErosion(img, kernel, dilationOrErosion=True):
    #A is image being processed, B is structuring element.
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]) #kernel to use
    xAxis1 = img.shape[0] #shape/size info for images/kernel
    yAxis1 = img.shape[1]
    xAxis2 = kernel.shape[0]
    yAxis2 = kernel.shape[1]
    copy = np.array(img)
    kerX = int(xAxis2/2)
    kerY = int(yAxis2/2)
    padded = np.zeros([xAxis1 + kerX, yAxis1 + kerY, 3]) #add padding to prevent out of bounds errors
    padded[kerX - 1 : -1, kerY - 1 : -1] = img #copy original image to new image
    kern = np.zeros([xAxis2, yAxis2, 3]) #make 3d array of 0s to convert 2D kernel
    for k in range(xAxis2):
        for l in range(yAxis2):
            num = kernel[k, l]
            kern[k, l] = [num, num, num] #convert kernel to RGB format
    kernel = kern
    for i in range(kerX, xAxis1):
        for j in range(kerY, yAxis1):#iterate through pixel in image
            region = padded[(i-kerX):(i+kerX+1), (j-kerY):(j+kerY+1)]
            #grab 3x3 area of pixels around current pixel
            val = 0
            bigPix = region * kernel
            if dilationOrErosion: #only difference is min/max, so I put both into 1 function
                bigPix = bigPix[bigPix != 0] #eliminate 0s from area
                #after union with kernel, since at least 1 pixel is guranteed to be 0
                #it makes all pixels in image black
                #print(bigPix)
                val = int(np.min(bigPix)) #apply dilation

            else:
                val = int(np.max(bigPix)) #apply erosion
            copy[i][j] = val #update pixel value
    return copy
def histogram_thresholding(image):#couldn't get this to compile, so i removed it
    # Compute the histogram of pixel intensities
    return 0
def clustering(img, clusters=3, iters=3): #attempted from Lecture 12 slide 11
    copy = np.array(img)
    xAxis = img.shape[0] #get general image info
    yAxis = img.shape[1]
    #Initialize: Pick K random points as cluster centers
    randK = np.random.randint(0, 255, size=(clusters, 3))
    #print(randK)
    for iter in range(iters): #amount of times to run k-means
        #your supposed to have k-means exit when it stops seeing significant changes
        # but I didn't get that to work
        tempIMG = np.array(img) #copy array
        z = 0
        for i in range(xAxis):
            for j in range(yAxis):
                #Assign data points to closest cluster center.
                pix = tempIMG[i][j]
                num = np.sqrt(np.sum(np.square(randK-pix),axis=1))
                z = np.argmin(num)
                tempIMG[i][j] = randK[z]
                #Change the cluster center to the average of its assigned points
        randK[z] = np.mean(tempIMG)
    return tempIMG


if __name__ == "__main__":  # go to main funciton
    #print(sys.argv)
    main()