'''
Created on Oct 20, 2017

@author: andre
'''
import glob
import pandas as pd
import numpy as np
import csv
import cv2
from pandas.io.pytables import IndexCol
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
import model
import tensorflow as tf

RandomState = 42


def LoadCSVFiles():
    dataFrame = pd.DataFrame() 
    trainingDataPath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project3/TrainingData*/*.csv"
    
    csvFiles = [file for file in glob.glob(trainingDataPath)]
    print(csvFiles)
    dfList = []
    names = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    for file in csvFiles:
        df = pd.read_csv(file, index_col = None, header = None, names = names)  
        dfList.append(df)
    dataFrame = pd.concat(dfList)
    print (dataFrame.describe())
    return dataFrame
 

def SplitDataInTrainingAndValidationSet(dataFrame):
    #trainingData, validationData = train_test_split(dataFrame, random_state = RandomState, test_size =0.2)
    trainingData, validationData = train_test_split(dataFrame, random_state = RandomState, test_size =0.2)
    return trainingData, validationData  



def DataPreparation(dataFrame, averagePerBin):
    condition = (dataFrame['Steering'] < 0.1) & ( dataFrame['Steering'] > -0.1)
    dataFrameZeroAngle = dataFrame[condition]
    fraction = averagePerBin/len(dataFrameZeroAngle) 
    print("Fraction = ", fraction)
    print ("Data entries removed ( -0.1 < steering < 0.1): ", len(dataFrameZeroAngle))
    dataFrameCleansed= dataFrame[~condition]
    print ("Data entries with |steering| > 0.1: ", len(dataFrameCleansed))    
    dataFrameZeroAngle = dataFrameZeroAngle.sample(frac=3*fraction,random_state = RandomState+3)
    print ("Data entries added back ( -0.1 < steering < 0.1): ", len(dataFrameZeroAngle))
    newDataFrame = pd.concat([dataFrameCleansed, dataFrameZeroAngle]) 
    newDataFrame = newDataFrame.sample(frac =1.0, random_state = RandomState+2)
    return newDataFrame

    
      
        
    
def RandomCamera(sample):
    centerLeftRight = ['Center', 'Left', 'Right']
    camera = centerLeftRight[np.random.randint(0,3)]
    
    path = sample[camera]
    #print(path)
    image = cv2.imread(path)
    angleCorrection = 0.25
    angle = sample['Steering']
    if(camera == 'Left'):
        angle += angleCorrection
    elif(camera == 'Right'):
        angle -= angleCorrection

    return image, angle

def RandomizeBrightness(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Use float value
    imageFloat = np.array(hsvImage, dtype = np.float64)
    randomBrightness = 0.5+np.random.uniform()
    imageFloat[:,:,2] = imageFloat[:,:,2]*randomBrightness
    imageFloat[:,:,2][imageFloat[:,:,2]>255]  = 255
    newImage = np.array(imageFloat, dtype = np.uint8)
    newImage = cv2.cvtColor(newImage,cv2.COLOR_HSV2BGR)
    return newImage


def RandomShift(image,angle,translationMax):
    widthTranslation = translationMax*np.random.uniform()-translationMax/2.0
    adjustedAngle = angle + widthTranslation/translationMax*2*.2
    heightTranslation = 40*np.random.uniform()-40/2
    
    translationMatrix = np.float32([[1,0,widthTranslation],[0,1,heightTranslation]])
    translatedImage = cv2.warpAffine(image,translationMatrix,(image.shape[1],image.shape[0]))

    return translatedImage, adjustedAngle


def RandomFlip(image, angle):
    flip = np.random.choice(2)

    if(flip):
        image = cv2.flip(image,1)
        angle = -angle
    
    return image, angle


def AddRandomShadow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    left = 320*np.random.uniform()
    top = 0
    bottom = 160
    right = 320*np.random.uniform()
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    
    shadowMask = 0*image[:,:,0]
    shadowMask[((X_m-top)*(right-left) -(bottom - top)*(Y_m-left) >=0)]=1

    if np.random.randint(2)==1:
        random_bright = .5
        condition1 = shadowMask==1
        condition2 = shadowMask==0
        if np.random.randint(2)==1:
            image[:,:,0][condition1] = image[:,:,0][condition1]*random_bright
        else:
            image[:,:,0][condition2] = image[:,:,0][condition2]*random_bright    

     
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return image


def DataAugmentation(sample):
    image, angle = RandomCamera(sample)
    flippedImage, flippedAngle = RandomFlip(image, angle)

    translationMax = 100.0
    shiftedImage, shiftedAngle = RandomShift(flippedImage, flippedAngle, translationMax)
    brightedImage = RandomizeBrightness(shiftedImage)
#    cv2.imshow('frame',translatedImage)
#    if cv2.waitKey(5000) & 0xFF == ord('q'):
#        x =1

    imageWithShadows = AddRandomShadow(brightedImage)

    return imageWithShadows, shiftedAngle 
            
'''
Data preprocessing for all images
'''

def ImagePreprocessing(inputImage, model):
    imageHeight = inputImage.shape[0] 
    imageWidth = inputImage.shape[1]
    newImage = inputImage[50:140,:]
    
    if(model[:-1] == "Nvidia"):
        newImage = cv2.resize(newImage,(200,66), interpolation = cv2.INTER_AREA)
    elif(model == "Other"):
        newImage = cv2.resize(newImage,(32,32), interpolation = cv2.INTER_AREA)
    
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2YUV)    
    newImage = newImage/127.5 - 1.0
    
    return newImage


def DataVisualization(labels, title):
    binNumber = 20
    plt.hist(labels, bins=binNumber, range = (-1, 1), label = "Including steering value 0.0")
    cleanedLabels = list(filter(lambda a: a != 0.0, labels))
    plt.hist(cleanedLabels, bins=binNumber, range = (-1, 1), label = "Excluding steering value 0.0")
    averagePerBin = len(labels)/binNumber
    plt.axhline(y= averagePerBin, xmin=-1, xmax=1, linewidth=2, color = 'b', label = "Average count per bin")
    plt.legend(loc='upper left', prop={'size': 10})
    plt.title(title)
    plt.show()
    return averagePerBin
    


'''
Data generator which yields the batches we need
'''
def DataGenerator(data, model, batchSize = 128):
    dataSize = len(data)
    images = []
    angles = []
    countZeroAngles = 0
    while True: # Loop forever so the generator never terminates
        while( len(images) < batchSize):
            sample = data.iloc[np.random.randint(dataSize-1), :]
            
            augmentedImage, augmentedAngle = DataAugmentation(sample)
            preprocessedImage = ImagePreprocessing(augmentedImage, model)
            images.append(preprocessedImage)
            angles.append(augmentedAngle)

            #cv2.imshow('frame',preprocessedImage)
            #if cv2.waitKey(1000) & 0xFF == ord('q'):
            #    x =1

        features = np.array(images)
        labels = np.array(angles)
        assert (len(features) == len(labels)  )
        
        yield sklearn.utils.shuffle(features, labels, random_state = RandomState+1)
        

