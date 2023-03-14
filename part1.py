# Brandon Mohan
# CMSC 630 Project Part 1
import copy
# used https://www.geeksforgeeks.org/working-csv-files-python/ to help open CSV file
# used https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/ to made 2D array
# used https://www.geeksforgeeks.org/writing-csv-files-in-python/
# used https://www.geeksforgeeks.org/python-removing-newline-character-from-string/
# used https://stackoverflow.com/questions/6483489/change-specific-rgb-color-pixels-to-another-color-in-image-file
# used https://www.geeksforgeeks.org/numpy-multiply-in-python/

import math
import sys #used to get inputFile
from PIL import Image #used to open Images
import matplotlib.pyplot as plot
import numpy as np #used to convert images to numbers for processing
import shlex #used to help parse input
import random
import time

if len(sys.argv) < 2: #make sure we have an input file
    print(sys.argv)
    print("Usage: python part1.py filename, RBG_Value, salt/Pepper strength, Gaussian Noise mean, Gaussian Noise sigma, mask size for boxfilter, pixel weight for boxfilter, mask for smoothing filter, weights for smoothing filter")
    sys.exit(1)

def main():  # main function
    startTime = time.time()#start time  to calculate image processing duraction
    endTime = time.time() #start time for calculating overall program duration
    #print(endTime)
    inputFile = sys.argv[1]
    numImages = 0
    names = [] #keep track of filenames
    cumHist = np.zeros(256) #make histogram for cumulative images
    timeCount = [] #tracks amount of time spent on each function
    with open(inputFile, 'r') as input:
        for line in input:
            startTime = time.time()
            imageTime = []
            numImages += 1
            parts = shlex.split(line) #break each line of input into array
            filename = parts[0]# name/path of image file
            names.append(filename)
            img = openImg(filename)
            #heres where we start knocking out the rubric
            monoImg = monoColor(img, parts[1]) #converts color images to selected single color spectrum
            imageTime.append(time.time()-startTime) #time to make monocolor image
            tempTime = time.time()
            #histograms
            hist = histogramCalc(monoImg) #make a histogram
            imageTime.append(time.time() - tempTime)#imageTime[-1])#time to make histogram
            #print(len(cumHist))
            #print(len(hist))
            tempTime = time.time()
            for num in range(0,255):
                cumHist[num] += hist[num] #add to cumulative histogram for getting the average of all images
            equalHisto = histoEqualization(monoImg, hist) #provide an equalized image of the histogram
            imageTime.append(time.time() - tempTime)#imageTime[-1]) #time to equalize histogram and make image
            hist2 = histogramCalc(equalHisto)#get histogram diagram of equalized histo
            #noise addition functions
            noiseTime = time.time() #
            salted = saltAndPepper(monoImg, int(parts[2]))#3000)
            imageTime.append(time.time() - noiseTime) #time to salt/pepper image
            noiseTime = time.time()  #
            gauNoise = gaussianNoise(monoImg, int(parts[3]), int(parts[4]))#0, 50)
            imageTime.append(time.time() - noiseTime)#imageTime[-1]) #time to add guassian noise to image
            #filtering options
            filterTime = time.time()
            box = boxFilter(monoImg, int(parts[5]), int(parts[6]))#5, 25) #linear filter
            imageTime.append(time.time() - filterTime)#time to apply box filter
            filterTime = time.time()
            toSmooth = parts[7].split("|")#np.matrix(parts[7].strip()).reshape(-1, 3)#np.fromstring(parts[7])#np.array([[int(j) for j in i.split('\t')] for i in parts[7].splitlines()])#np.fromstring(parts[7], )#[[3, 5, 3], [5, 8, 5], [3, 5, 3]]) #median filter
            smooth = []
            for one in toSmooth: #taking a mask as input
                smo = []
                two = one.split(",")
                for three in two:
                    #print("t, ", three)
                    smo.append(int(three))
                smooth.append(smo)
            smooth = np.array(smooth) #median filter?
            smoothed = smoothingFilter(monoImg, smooth, (int(parts[8])/int(parts[9])))#1/40)
            imageTime.append(time.time() - filterTime)#imageTime[-1]) #time to apply smoothing filter

            #display all the images - comment these out if you want
            print(line)
            Image.fromarray(monoImg, 'RGB').show()
            #print(hist)
            plot.title("Histogram for " + filename)
            plot.bar(range(256), hist)
            plot.show()
            #print(equalHisto)
            #print(hist2)
            plot.title("Equalized Histogram for " + filename)
            plot.bar(range(256), hist2)
            plot.show()
            Image.fromarray(equalHisto, 'RGB').show()
            Image.fromarray(salted, 'RGB').show()
            Image.fromarray(gauNoise, 'RGB').show()
            Image.fromarray(box, 'RGB').show()
            Image.fromarray(smoothed, 'RGB').show()
            timeCount.append(imageTime)

    for images in range(numImages): #for each image processed
        sumImg = 0
        for tim in timeCount[images]:#print amount of time spend on each image
            sumImg += tim
        print(names[images])
        print(timeCount[images])
        print("Total Image Time: ", sumImg)
    cumHist = np.array(cumHist) / numImages
    #print("Average Histogram", cumHist/numImages)
    histList = list(cumHist)
    #print(histList)
    plot.title("Average Histogram for program")
    plot.bar(range(256), cumHist)
    plot.show()
    print("TOTAL TIME for program: ", time.time() - endTime)
    return 0

#def fixArgs(cmd):
#    args = []
#    length = len(cmd)
#    for i in range(8):
#        if length > i:
#            args.append(cmd[i])
#        else:


def openImg(name):#save a few lines of code in main, was originally going to structure code differently
    raw = Image.open(name)
    rgb = raw.convert('RGB')
    img = np.array(rgb)
    return img

def monoColor(img, color='b'):#picks RGB value and converts image to gray
    lower = color.lower()
    xAxis = img.shape[0]
    yAxis = img.shape[1]
    r, g, b = 0, 0, 0  # Original value
    colorIndex = -1
    comparator = np.zeros(3)
    if 'r' in lower: #if R, then use Red values
        colorIndex = 0
        comparator[0] = 1
    if 'g' in lower: #if green, then use green values
        colorIndex = 1
        comparator[1] = 1
    if 'b' in lower: #if blue/no arguments, then use blue values
        colorIndex = 2
        comparator[2] = 1

    copyImg = np.zeros_like(img)
    index = 0
    for i in range(xAxis):
        for j in range(yAxis):#loop and change all pixels to only use the chosen RGB value
            pixel = img[i][j] #put chosen value into new image
            monoPixel = np.multiply(pixel, comparator)
            copyImg[i][j] = monoPixel[colorIndex]
            #print(monoPixel)
    return copyImg

def histogramCalc(imgToHist): #following the sudocode in lecture 3 slide 12/32
    xAxis = imgToHist.shape[0] #get size of image
    yAxis = imgToHist.shape[1]
    dictionary = {}
    for d in range(256):
        dictionary[d] = 0 #make a dict for all possible 256 pixel values
    for x in range(xAxis):
        for y in range(yAxis):
            dictionary[imgToHist[x][y][0]] += 1
            #for pixelVal in range(256): #from slide sudocode, i made some improvements
                #if imgToHist[x][y][0] == pixelVal:
                    #histogram[pixelVal+1] += 1
                    #break
    keyVals = dictionary.values()
    data = list(keyVals) #convert dict to np array
    histogram = np.array(data)
    #plot.bar(range(256), data)
    #plot.show()
    return histogram

def histoEqualization(pic, histo):
    #created using https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
    cumulative = []
    xAxis = pic.shape[0] #get size of image
    yAxis = pic.shape[1]
    p = histo/(xAxis*yAxis)
    img2 = pic.copy()
    sum = 0
    #print("HISTO", histo)
    for h in p:
        sum += h
        cumulative.append(sum)
    #print("SUM", sum)
    for i in range(0, xAxis):
        for j in range(0, yAxis):
            intensity = pic[i][j][0]
            img2[i][j] = int(cumulative[intensity] * 255)
    #newHisto = histogramCalc(img2)
    return img2

def saltAndPepper(image, strength):
    img2 = image.copy()
    for x in range(image.shape[0]): #loop through each pixel in image
        for y in range(image.shape[1]):
            if random.randint(0, 1000000) < strength: #strength/million chance
                #if unlucky, then we ruin pixel
                coinflip = random.randint(0, 1)
                if coinflip == 0:
                    img2[x][y] = 0 #the pepper
                else:
                    img2[x][y] = 255 #the salt

    return img2

def gaussianNoise(pic, mean, sigma): #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    copyImg = np.zeros_like(pic)
    xAxis = pic.shape[0]
    yAxis = pic.shape[1]
    ary = []
    for x in range(xAxis): #loop through each pixel
        for y in range(yAxis):
            gaussian = np.random.normal(mean, sigma, 1) #normal gaussian distribution
            temp = pic[x][y]
            temp = temp + gaussian #apply to each pixel
            temp = np.asarray(temp, dtype='int')
            temp = temp.clip(0, 255)
            copyImg[x][y] = temp #put new pixel in new image

    return copyImg

def boxFilter(pict, maskSize, pixelWeight): #made from lecture 6 slide 8/34
    #padded it
    w = pict.shape[0]
    h = pict.shape[1]
    floor = int(maskSize/2) #casting is the same as floor function lol
    ceil = round(maskSize/2)#rounding is good enough
    copyImg = np.zeros_like(pict)

    for v in range(floor, h-ceil):
        for u in range(floor, w-ceil): #for each pixel in valid range
            sum = 0
            count = 0
            for j in range(-1* floor, ceil+1):
                for i in range(-1*floor, ceil+1):#apply the corresponding pixel in the mask
                    p = pict[u+i][v+j][0]
                    sum += p
                    count += 1
            q = sum/pixelWeight
            copyImg[u][v] = round(q)
    return copyImg

def smoothingFilter(pict, mask, multiplier=1): #sudocode taken from lecture 6 slide 10/34
    w = pict.shape[0]
    h = pict.shape[1]
    maskSize = mask.shape[0]#supposed to be a square, so we only need 1 dimension
    maskSum = np.sum(mask)*multiplier
    #print(maskSum)
    #print(maskSize)
    floor = int(maskSize/2)
    ceil = round(maskSize/2)
    copyImg = np.zeros_like(pict)
#same loops as box filter
    for v in range(floor, h-ceil):
        for u in range(floor, w-ceil): #for each pixel in image
            sum = 0
            count = 0
            for j in range(-1* floor, ceil): #grab corresponding pixel in the mask
                for i in range(-1*floor, ceil):
                    p = pict[u+i][v+j][0]
                    c = mask[j+1][i+1]
                    sum += round(p * c * multiplier)
                    count += 1
            q = round(sum)
            #print("COUNT: ", count)
            #print("q ", q)
            copyImg[u][v] = round(q)
    return copyImg


if __name__ == "__main__":  # go to main funciton
    #print(sys.argv)
    main()