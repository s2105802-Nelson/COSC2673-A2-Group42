## This is a python file for reusable functions that are specific to the COSC2673 Assignment 2.
import pandas as pd
import numpy as np
import cv2

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def loadImagesToFlattened(df, imageCol):
    # loop through the images, load them and add the to the list
    images = []
    for imageName in df[imageCol]:
        img = cv2.imread(imageName)
        images.append(img)

    # After loading, Each image is an array of 3-element arrays, corresponding to the RGB values. Flatten each image so that each image is a single array of numbers 
    imagesFlat = [np.reshape(img, (-1,)) for img in images]
    return imagesFlat

def getClassificationROC(predictor_name, set_name, y_true, y_pred, num_classes=2, y_pred_scores=None):
    if num_classes <= 1:
        print("Error: Number of label classes must be 2 or more")
        return -1
    elif num_classes == 2:
        # Do a Binary Classification ROC Curve

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        plt.figure(1)
        plt.plot([0, 1], [0, 1])
        plt.plot(fpr, tpr, label="CNN(area = {:.3f})".format(roc_auc))
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(predictor_name + " " + set_name + " Set ROC curve")
        plt.legend(loc="best")
        plt.show()

        print("ROC (Area): " + str(roc_auc))
        return roc_auc
    else:
        # Do a multi-class, which will show a ROC Curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        list_roc_auc = []
        
        # first binarize the labels (converting from a number to a list of booleans)        
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Loop through each class, Compute ROC curve and ROC area for each class
        for i in range(num_classes):
            # Get a list of the scores for this class
            list_scores = []
            for j in range(len(y_pred_scores)):
                list_scores.append(y_pred_scores[j][i])
                
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], list_scores)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(num_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(predictor_name + " " + set_name + " Set ROC for class number " + str(i))
            plt.legend(loc="lower right")
            plt.show()  

            list_roc_auc.append(roc_auc[i])

        # Average the area under curve for each, print and return
        roc_auc_mean = np.mean(list_roc_auc)     
        print("Mean ROC (Area): " + str(roc_auc_mean))     
        return roc_auc_mean


        