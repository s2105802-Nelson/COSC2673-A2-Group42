################################
### Utility file for Pandas datafame manipulation helper functions
################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# For a dataframe and a list of columns, return a copy of the df with those columns dropped
def getFeaturesWithoutCols(df1, listColsToExclude):
  dfContext = df1.copy()
  for i in range(len(listColsToExclude)):
    del dfContext[listColsToExclude[i]]
  return dfContext


def convertBoolColToIntHelper(input):
  if input is None or input.strip() == "":
    return 0
  inputLower = input.lower().strip()
  if inputLower == "y" or inputLower == "yes" or inputLower == "1" or inputLower == "t" or inputLower == "true":
     return 1
  else:
     return 0


def convertBoolColToInt(dfInput, columnName):
  return dfInput.apply(lambda x: convertBoolColToIntHelper(x[columnName]), axis=1)


# We often want to have 3 splits, Train, Validation and Test. The easiest way is to just do
# 2 splits. Use this function for that, where this assumes that the validation and test splits are 
# half of the remainder from the train split.
# so the default is 60%/20%/20%
# This version takes in a dataframe and splits into 3
def train_val_test_split(df, train_size=0.6, random_state=42):
  dfTrain, dfValTest = train_test_split(df, test_size=(1-train_size), random_state=random_state)
  dfVal, dfTest = train_test_split(dfValTest, test_size=0.5, random_state=random_state)
  return dfTrain, dfVal, dfTest


# This version is when you have 2 sets to split, e.g. you have already split into features and targets
def train_val_test_split_2(dfFeatures, dfTargets, train_size=0.6, random_state=42):
  dfTrainFeat, dfValTestFeat, dfTrainTargets, dfValTestTargets = train_test_split(
      dfFeatures, dfTargets, test_size=(1-train_size), random_state=random_state)
  dfValFeat, dfTestFeat, dfValTargets, dfTestTargets = train_test_split(
      dfValTestFeat, dfValTestTargets, test_size=0.5, random_state=random_state)
  return dfTrainFeat, dfValFeat, dfTestFeat, dfTrainTargets, dfValTargets, dfTestTargets


# This version is when you have 3 sets to split, e.g. you have already split into ids, features and targets
def train_val_test_split_3(dfIds, dfFeatures, dfTargets, train_size=0.6, random_state=42):
  dfTrainIds, dfValTestIds, dfTrainFeat, dfValTestFeat, dfTrainTargets, dfValTestTargets = train_test_split(
      dfIds, dfFeatures, dfTargets, test_size=(1-train_size), random_state=random_state)
  dfValIds, dfTestIds, dfValFeat, dfTestFeat, dfValTargets, dfTestTargets = train_test_split(
      dfValTestIds, dfValTestFeat, dfValTestTargets, test_size=0.5, random_state=random_state)
  return dfTrainIds, dfValIds, dfTestIds, dfTrainFeat, dfValFeat, dfTestFeat, dfTrainTargets, dfValTargets, dfTestTargets


################################
### START: One Hot Encoding Helper Functions
################################

# Looking for a basic one-hot-encoding function? Do:
# df = pd.get_dummies(data=df, columns=["Country"])

# Given two data frames of the same format (assuming that they have been Train/Validate split), and a column name
# Will do one-hot-encoding using the pandas get_dummies() function, ensuring the same format on both. It does this by
# combining the two sets, do the one hot encoding, then splitting it back up again
def getDummiesForSplitSets(df1, df2, colName, dropOriginalCol=True):
  df1["getDummiesForSplitSets_Source"] = "df1"
  df2["getDummiesForSplitSets_Source"] = "df2"

  # Convert the col to a categorical through one hot encoding. Do this for both test and vali
  # and make sure the one hot encoding is the same through columns. We do this by
  # joining them together, do the one hot encoding, then separate them again.

  df_combined = df1.append(df2)
  # print(df_combined.shape)
  # df_combined.head()

  # Do the one hot encoding
  df_combined = pd.concat([df_combined, pd.get_dummies(df_combined[colName], prefix=colName, prefix_sep="_")], axis=1)

  # Format the columns, removing common punctuation
  df_combined.columns = df_combined.columns.str.replace(" ", "").str.replace("/", "").str.replace("-", "") \
    .str.replace(":", "").str.replace(";", "").str.replace("'", "").str.replace("\"", "").str.replace("&", "") \
    .str.replace("$", "").str.replace("@", "").str.replace(",", "").str.replace(".", "").str.replace("?", "").str.replace("!", "")

  # In general, after one hot encoding, you want to drop the original column name, but use the param to stop this if wanted
  if dropOriginalCol:
    df_combined = df_combined.drop(columns=colName)

  # now split up the data again and drop the source
  df1 = df_combined[df_combined["getDummiesForSplitSets_Source"] == "df1"]
  df2 = df_combined[df_combined["getDummiesForSplitSets_Source"] == "df2"]
  df1 = df1.drop(columns="getDummiesForSplitSets_Source")
  df2 = df2.drop(columns="getDummiesForSplitSets_Source")

  # delete the combined
  del df_combined  

  return df1, df2


# Do the one hot encoding just for one dataset (for the test data). Basically replicate the above logic, but just on one dataframe
def getDummiesForSingleSet(df1, colName, dropOriginalCol=True):

  # Do the one hot encoding
  df1 = pd.concat([df1, pd.get_dummies(df1[colName], prefix=colName, prefix_sep="_")], axis=1)

  # Format the columns, removing common punctuation
  df1.columns = df1.columns.str.replace(" ", "").str.replace("/", "").str.replace("-", "") \
    .str.replace(":", "").str.replace(";", "").str.replace("'", "").str.replace("\"", "").str.replace("&", "") \
    .str.replace("$", "").str.replace("@", "").str.replace(",", "").str.replace(".", "").str.replace("?", "").str.replace("!", "")

  # In general, after one hot encoding, you want to drop the original column name, but use the param to stop this if wanted
  if dropOriginalCol:
    df1 = df1.drop(columns=colName)

  return df1  


# Found on complete runs, that doing final prediction on the Test data failed with features not matching.
# This one hot encoding process needs to go across all 3 datasets to ensure they all have consistent columns for
# Across the three sets
def getDummiesForTripleSets(df1, df2, df3, colName, dropOriginalCol=True):
  df1["getDummiesForTripleSets_Source"] = "df1"
  df2["getDummiesForTripleSets_Source"] = "df2"
  df3["getDummiesForTripleSets_Source"] = "df3"

  # Convert the col to a categorical through one hot encoding. Do this for both test and vali
  # and make sure the one hot encoding is the same through columns. We do this by
  # joining them together, do the one hot encoding, then separate them again.

  df_combined = df1.append(df2).append(df3)
  print(df_combined.shape)
  df_combined.head()

  # Do the one hot encoding
  df_combined = pd.concat([df_combined, pd.get_dummies(df_combined[colName], prefix=colName, prefix_sep="_")], axis=1)

  # Format the columns, removing common punctuation
  df_combined.columns = df_combined.columns.str.replace(" ", "").str.replace("/", "").str.replace("-", "") \
    .str.replace(":", "").str.replace(";", "").str.replace("'", "").str.replace("\"", "").str.replace("&", "") \
    .str.replace("$", "").str.replace("@", "").str.replace(",", "").str.replace(".", "").str.replace("?", "").str.replace("!", "")

  # In general, after one hot encoding, you want to drop the original column name, but use the param to stop this if wanted
  if dropOriginalCol:
    df_combined = df_combined.drop(columns=colName)

  # now split up the data again and drop the source
  df1 = df_combined[df_combined["getDummiesForTripleSets_Source"] == "df1"]
  df2 = df_combined[df_combined["getDummiesForTripleSets_Source"] == "df2"]
  df3 = df_combined[df_combined["getDummiesForTripleSets_Source"] == "df3"]
  df1 = df1.drop(columns="getDummiesForTripleSets_Source")
  df2 = df2.drop(columns="getDummiesForTripleSets_Source")
  df3 = df3.drop(columns="getDummiesForTripleSets_Source")

  # delete the combined
  del df_combined  

  return df1, df2, df3  

  
################################
### END: One Hot Encoding Helper Functions
################################
