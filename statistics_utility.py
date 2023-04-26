from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from itertools import chain
import seaborn as sns


################################
### START: Basic Stats and Data Explore Helper Functions
################################

def df_get_info(df, dfName, dfNameHeader=True):
    if dfNameHeader:
        print("Dataframe Name: " + dfName)
    print("Dataframe Shape (rows / columns): " + str(df.shape))
    print("-----------------------------\n")
    print("dataframe.info()")
    print("--------------------")
    print(df.info())


def df_get_describe(df, dfName, dfNameHeader=True):
    if dfNameHeader:
        print("Dataframe Name: " + dfName)

    print("dataframe.describe()")    
    if df.shape[1] < 10:
      print("--------------------")
      print(df.describe())
    else:
      # More than 10 columns, report on the cols 10 at a time
      colCount = df.shape[1]
      minCol = 0
      maxCol = 9

      # Loop though, describing columns 10 at a time
      while maxCol < colCount:
        print("Describe Cols " + str(minCol) + " - " + str(maxCol))
        print(print(df.iloc[:, range(minCol, maxCol)].describe()))
        minCol = maxCol + 1
        maxCol = maxCol + 10        
        # In this iteration, we will have reached the end of the columns
        if maxCol > colCount:
          maxCol = colCount
          # the loop will end in the next iteration, if there are a last set of final columns, print their details
          if maxCol > minCol:
            print("Describe Cols " + str(minCol) + " - " + str(maxCol))
            print(print(df.iloc[:, range(minCol, maxCol)].describe()))
            minCol = maxCol + 1
            maxCol = maxCol + 10      


def df_get_uniques(df, dfName, dfNameHeader=True):  
    if dfNameHeader:
        print("Dataframe Name: " + dfName)   
    print("Dataframe Column Unique Values")
    print("--------------------")
    for col in df:
      print("Column '" + col + "' - Unique value count: " + str(len(df[col].unique())))
      if len(df[col].unique()) <= 20:
          print("   Column '" + col + "' values:")
          print("   " + str(df[col].unique()))


### Pass in a dataframe. Will output the basic data information on the dataframe
def df_basic_data_info(df, dfName, largeTextWarning=True):
    df_get_info(df, dfName, True)
    print("-----------------------------\n")
    df_get_describe(df, dfName, False)
    print("-----------------------------\n")
    df_get_uniques(df, dfName, False)
    print("-----------------------------\n")
    if largeTextWarning:
      print("If full information does not fit into the output window, get the info separately with the following:")
      print("statsutil.df_get_info(df, \"" + dfName + "\")")
      print("statsutil.df_get_describe(df, \"" + dfName + "\")")
      print("statsutil.df_get_uniques(df, \"" + dfName + "\")")


### Pass in a dataframe. Will output a grid of basic distributions histogram plots for every column
def df_col_distributions_all(dataframe, color="b"):
  plt.figure(figsize=(22,22))
  for i, col in enumerate(dataframe.columns):
    plt.subplot(4,5,i+1)
    plt.hist(dataframe[col], alpha=0.3, color=color, density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


### Looking to do a matrix of scatterplots next? no need for a function in here, just do
# pd.plotting.scatter_matrix(dataFrame)


### Pass in a dataframe. Will output the correlation heatmap matrix between all the columns
def df_correlation_matrix(dataframe):
    f, ax = plt.subplots(figsize=(11, 9))
    corr = dataframe.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    );


### Works for a Series (dataframe column), not testing with a list or np array yet
### A reusable function that examines the statistics, distribution and outlier information for a data column
### e.g usage: 
###   print_stats_and_outliers(dfBikesHour["hr"], "hr", "Hour", True, 24, False)
def print_stats_and_outliers(variableData, variableName, variableNiceName, showPlots=True, histBins=-1, histNoZeroes=True):
  variableRecordCount = len(variableData)
  variableMin = np.min(variableData)
  variableMax = np.max(variableData)
  variableMissing = variableData.isnull().sum()
  variableMean = np.mean(variableData)
  variableMedian = np.median(variableData)
  variableSD = np.std(variableData)
  variableQ1 = np.nanquantile(variableData, 0.25)
  variableQ3 = np.nanquantile(variableData, .75)
  variableIQR = variableQ3 - variableQ1
  variableLowerFence = variableQ1 - (1.5 * variableIQR)
  variableUpperFence = variableQ3 + (1.5 * variableIQR)  
    
  variableZeros = len(variableData[variableData == 0])
  variableLowerOutliers = len(variableData[variableData < variableLowerFence])
  variableUpperOutliers = len(variableData[variableData > variableUpperFence])

  print("Basic Summary Statistics of " + variableNiceName + " - " + variableName + ":")
  print("  Records: " + str(variableRecordCount))
  print("  Null Records: " + str(variableMissing) + "   Zero Records: " + str(variableZeros))
  print("  Min: " + str(variableMin) + "   Max: " + str(variableMax))
  print("  Mean: " + str(variableMean) + "   Median: " + str(variableMedian))
  print("  Standard Deviation: " + str(variableSD))
  print("Quantile Statistics:")
  print("  Q1: " + str(variableQ1) + "   Q3: " + str(variableQ3) + "   IQR: " + str(variableIQR))
  print("  Lower Fence: " + str(variableLowerFence) + "   Upper Fence: " + str(variableUpperFence))
  print("Outlier Counts:")
  print("  Lower Outliers: " + str(variableLowerOutliers) + "   Upper Outliers: " + str(variableUpperOutliers))  
  print("  Total Outliers: " + str(variableLowerOutliers + variableUpperOutliers))

  if showPlots:
    if histBins == -1:
      # if not set, set the default number of bins for the histogram to 13
      histBins = 13

    # look at a histogram and a boxplot of this data, to view distribution and look for possible outliers    
    variableData.plot(kind='hist',bins=histBins)
    plt.title(variableNiceName + " Histogram")
    plt.xlabel(variableNiceName)    

    fig2, ax2 = plt.subplots()
    ax2.set_title(variableNiceName + " Boxplot")
    plt.boxplot(variableData, labels=[variableNiceName])
    plt.show()

    if histNoZeroes:
      # Do another histogram of the data, filtering out zeroes
      variableData[variableData > 0].plot(kind='hist',bins=histBins)
      plt.title(variableNiceName + " Histogram No Zeroes")
      plt.xlabel(variableNiceName)
      plt.show()

  # print(paste("Number of Detected Outliers: ", length(bp$out), sep=" "))  


# For a given column in a dataframe, do a basic IQR based calculation to find a number of possible outliers. This is a good way to determine
# whether we should generate a boxplot
def get_outlier_info_for_col(df, colName):
  variableData = df[colName]
  variableQ1 = np.nanquantile(variableData, 0.25)
  variableQ3 = np.nanquantile(variableData, .75)
  variableIQR = variableQ3 - variableQ1
  variableLowerFence = variableQ1 - (1.5 * variableIQR)
  variableUpperFence = variableQ3 + (1.5 * variableIQR) 
  variableLowerOutliers = len(variableData[variableData < variableLowerFence])
  variableUpperOutliers = len(variableData[variableData > variableUpperFence]) 
 
  return variableLowerFence, variableUpperFence, variableLowerOutliers, variableUpperOutliers, variableLowerOutliers + variableUpperOutliers


# For all columns in a dataframe, do a basic IQR based calculation to find a number of possible outliers. This is a good way to determine
# whether we should generate a boxplot for any of the columns
# Only pass in a dataframe of numerical columns
def df_outlier_info(df, printDF=False):
  print("Dataframe shape: " + str(df.shape))

  colNames = df.columns
  dfOutliers = pd.DataFrame(columns=["Column", "LowerFence", "UpperFence", "LowerOutliers", "UpperOutliers", "TotalOutliers"])

  for col in colNames:
    lowerFence, upperFence, lowerOutliers, upperOutliers, totalOutliers = get_outlier_info_for_col(df, col)
    newRow = { "Column": col,"LowerFence" : lowerFence, "UpperFence" : upperFence, "LowerOutliers" : lowerOutliers, "UpperOutliers": upperOutliers, "TotalOutliers": totalOutliers }
    dfOutliers = dfOutliers.append(newRow, ignore_index=True)

  if printDF:
    dfOutliers.head(len(colNames))

  return dfOutliers



# Calculate upper and lower fence, then null out entries in the column according to those fences
def auto_rm_outliers(dataframe, column_name):  
  dfCopy = dataframe.copy()
  
  variableQ1 = np.nanquantile(dfCopy[column_name], 0.25)
  variableQ3 = np.nanquantile(dfCopy[column_name], .75)
  variableIQR = variableQ3 - variableQ1
  variableLowerFence = variableQ1 - (1.5 * variableIQR)
  variableUpperFence = variableQ3 + (1.5 * variableIQR)  

  dfCopy = dfCopy[(dfCopy[column_name] >= variableLowerFence) & (dfCopy[column_name] <= variableUpperFence)]
  return dfCopy


def null_out_zeros(dataframe, column_name):  
  dfCopy = dataframe.copy()  
  dfCopy[column_name] = dfCopy[column_name].apply(lambda x: np.NaN if x == 0 else x)
  return dfCopy

      
################################
### END: Basic Statisitcs Helper Functions
################################


################################
### START: Confusion Matrix Helpers
################################

# Returns a string report on a confusion matrix, only for a binary categorisation (i.e. a 2x2 matrix)
# pass in a result from a confusion_matrix() 
def confusionMatrixBinaryReport(cm, showFullStats=False):
  cmReport = "Confusion Matrix:\n"
  cmReport += str(cm) + "\n\n"

  tp = cm[0, 0]
  tn = cm[1, 1]
  fp = cm[1, 0]
  fn = cm[0, 1]
  total = tp + tn + fp + fn

  cmReport += "  Total Records: " + str(total) + "\n"
  cmReport += "  Correct Predictions: " + str(tp + tn) + "\n"
  cmReport += "  Incorrect Predictions: " + str(fp + fn) + "\n"  
  cmReport += "  Accuracy Rate (Proportion of Correct predictions): " + str((tp + tn) / total) + "\n\n"

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  cmReport += "  Precision (When predicting Positive, how many were actually Positive): " + str(precision) + "\n"  
  cmReport += "  Recall/Sensitivity (How many predicted Positive were actually Positive): " + str(recall) + "\n"

  if showFullStats:
    cmReport += "\n  Negative Predictive Value (When predicting Negative, how many were actually Negative): " + str(tn / (tn + fn)) + "\n"
    cmReport += "  Specificity (How many predicted Negative were actually Negative): " + str(tn / (tn + fp)) + "\n"
    cmReport += "  F1-Score ( F = 2 * ([Precision * Recall] / [Precision + Recall]) ): " + str(2 * ( (precision*recall) / (precision+recall))) + "\n"

  return cmReport


################################
### END: Confusion Matrix Helpers
################################


################################
### START: Data Transformation Helpers
################################


def doLog10Transform(val):
  if np.isnan(val):
    return np.NaN
  elif val == 0:
    return 0    
  else:
    return np.log10(val)


def log10TransformDfCol(df, colName):
  df[colName] = df.apply(lambda x: doLog10Transform(x[colName]), axis=1)
  return df


def doPowerTransform(val, order):
  if np.isnan(val):
    return np.NaN
  elif val == 0:
    return 0    
  else:
    return val ^ order


def powerTransformDfCol(df, colName, order=2):
  df[colName] = df.apply(lambda x: doPowerTransform(x[colName], order), axis=1)
  return df


def scaleStandardSingle(dataframeCol):
  scaler = StandardScaler()
  dataframeCol = scaler.fit_transform(dataframeCol)
  return dataframeCol


def scaleMinMaxSingle(dataframeCol):
  scaler = MinMaxScaler()
  dataframeCol = scaler.fit_transform(dataframeCol)
  return dataframeCol


def scaleStandardMultiple(df, feature_cols):
  scaler = StandardScaler()
  dfScaled = scaler.fit_transform(df[feature_cols])
  return dfScaled    
      

def scaleMinMaxMultiple(df, feature_cols):
  scaler = MinMaxScaler()
  dfScaled = scaler.fit_transform(df[feature_cols])
  return dfScaled    


# Calculate the r2_adjusted score. This is a version of the r-squared that takes into account the degrees of freedom
# this is a better way of comparing two models where the number of features are different
# Formula: AdjustedR2 = 1 – [(1-R2)*(n-1)/(n-k-1)]
def r2_adjusted_score(dfTarget, dfPredicted, dfFeatures):
  r2 = r2_score(dfTarget, dfPredicted)
  r2ajd = 1 - (1-r2) * (len(dfTarget)-1)/(len(dfTarget)-dfFeatures.shape[1]-1)
  return r2ajd
      
################################
### END: Data Transformation Helpers
################################


################################
### START: NDCG For Ranking Helper Functions
################################


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        #print(f"dcg_at_k {str(r.size)}")
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    #print(f"ndcg_at_k {str(idcg)}")
    if not idcg:
        return 1         

    #rint(f"ndcg_at_k returning {str(dcg_at_k(r, k))} / {str(idcg)} = {str(dcg_at_k(r, k) / idcg)}")
    return dcg_at_k(r, k) / idcg


def ndcg_for_dataset(df, k=10):  
  relevances = df.groupby("Query ID")
  ndcg = relevances.apply(lambda x: ndcg_at_k(x["Label"], k)).mean()
  return ndcg

################################
### END: NDCG For Ranking Helper Functions
################################

