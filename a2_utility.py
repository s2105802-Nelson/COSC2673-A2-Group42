## This is a python file for reusable functions that are specific to the COSC2673 Assignment 2.
import pandas as pd
import numpy as np
import cv2

def loadImagesToFlattened(df, imageCol):
    # loop through the images, load them and add the to the list
    images = []
    for imageName in df[imageCol]:
        img = cv2.imread(imageName)
        images.append(img)

    # After loading, Each image is an array of 3-element arrays, corresponding to the RGB values. Flatten each image so that each image is a single array of numbers 
    imagesFlat = [np.reshape(img, (-1,)) for img in images]
    return imagesFlat