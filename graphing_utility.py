import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Basic Scatterplot between two features
def graphBasicScatter(dfInput, xCol, yCol, title, xLabel, yLabel):    
    plt.title(title)
    xaxis = dfInput[xCol]
    yaxis = dfInput[yCol]

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    # Output the scatter plot. Also, print the correlation coefficient
    pltOutput = plt.scatter(xaxis, yaxis)
    print("Correlation Coefficient: " + str(dfInput[xCol].corr(dfInput[yCol])))
    
    return pltOutput 

# Basic Line Graph with two features
def graphBasicSingleLine(dfInput, xCol, yCol, title, xLabel, yLabel, useMarkers=False):
    plt.title(title)
    xaxis = dfInput[xCol]
    yaxis = dfInput[yCol]

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    marker = ''
    if useMarkers:
        marker = '.'

    pltOutput = plt.plot(xaxis, yaxis, marker=marker)
    return pltOutput 
    
# Basic Line Graph with two series. Specify a single X feature and then two features to plot as two lines
def graphBasicTwoSeries(dfInput, xCol, series1Col, series2Col, title, xLabel, series1Label, series2Label, useMarkers=False, showLegend=True):
    plt.title(title)
    xaxis = dfInput[xCol]
    series1 = dfInput[series1Col]
    series2 = dfInput[series2Col]

    plt.xlabel(xLabel)
    marker1 = ''
    marker2 = ''
    if useMarkers:
        marker1 = '.'
        marker2 = 'x'

    plt.plot(xaxis, series1, c='#0000FF', marker=marker1, markersize=8, label=series1Label) 
    plt.plot(xaxis, series2, c='#FF0000', marker=marker2, markersize=8, label=series2Label) 

    if showLegend:
        plt.legend()

    return plt 
    

# Basic Line Graph with three series. Specify a single X feature and then three features to plot as two lines    
def graphBasicThreeSeries(dfInput, xCol, series1Col, series2Col, series3Col, title, xLabel, series1Label, series2Label, series3Label, useMarkers=False, showLegend=True):
    plt.title(title)
    xaxis = dfInput[xCol]
    series1 = dfInput[series1Col]
    series2 = dfInput[series2Col]
    series3 = dfInput[series3Col]

    plt.xlabel(xLabel)
    marker1 = ''
    marker2 = ''
    marker3 = ''
    if useMarkers:
        marker1 = '.'
        marker2 = 'x'
        marker3 = 'o'

    plt.plot(xaxis, series1, c='#0000FF', marker=marker1, markersize=8, label=series1Label) 
    plt.plot(xaxis, series2, c='#FF0000', marker=marker2, markersize=8, label=series2Label) 
    plt.plot(xaxis, series3, c='#2e5739', marker=marker3, markersize=8, label=series3Label) 

    if showLegend:
        plt.legend()

    return plt 
       

# Basic bar plot of counts of values
# Do a basic Group By Count on the dataframe then return a basic matplotlib bar plot (which you can do a plt.show() on)
def graphBasicBarDistribution(dfInput, categoricalCol):
    # Do a Group By Count on the categorical values
    dfGroupByCount = dfInput.groupby(categoricalCol).agg(ResultCount = (categoricalCol, 'size')).sort_values(categoricalCol, ascending = True)
    dfGroupByCount = dfGroupByCount.reset_index(level=0)

    # Create the distribution plot as a bar plot
    plt.title(categoricalCol + " - Count of Values")
    plt.bar(dfGroupByCount[categoricalCol], dfGroupByCount["ResultCount"], label='ResultCount')
    plt.legend()
    return plt


# Basic Bar plot to show how many values in a text feature have text or are empty/NA
# For a Text column: Visualise how many records have text, are empty strings, or are NA
def graphBasicEmptyTextDistribution(dfInput, textCol):
    countNA = dfInput[dfInput[textCol].isna()].shape[0]
    countEmpty = dfInput[dfInput[textCol].str.strip() == ""].shape[0]
    countHasText = dfInput[dfInput[textCol].str.strip() != ""].shape[0]

    dfTextData = [ ["Is NA", countNA], ["No Text", countEmpty], ["Has Text", countHasText] ]
    dfTextCounts = pd.DataFrame(data = dfTextData, columns=["TextStatus", "Count"])
    
    plt.title(textCol + " - Count by Text Status")
    bars = plt.bar(dfTextCounts["TextStatus"], dfTextCounts["Count"], label='Count')
    plt.bar_label(bars)
    return plt    


# For a general data column: Visualise with a Bar Plot how many records are NA or not 
def graphBasicNADistribution(dfInput, targetCol):
    countNA = dfInput[dfInput[targetCol].isna()].shape[0]
    countNotNA = dfInput[dfInput[targetCol].notna()].shape[0]

    dfData = [ ["Is NA", countNA], ["Not NA", countNotNA] ]
    dfNACounts = pd.DataFrame(data = dfData, columns=["NA Status", "Count"])
    
    plt.title(targetCol + " - Count by Text Status")
    bars = plt.bar(dfNACounts["NA Status"], dfNACounts["Count"], label='Count')
    plt.bar_label(bars)
    return plt        


# Visualise a distribution as a bar plot for a particular basic boolean condition
def graphBasicConditionDistribution(dfInput, mskCondition, conditionTitle=""):
    countYes = dfInput[mskCondition].shape[0]
    countNo = dfInput[~mskCondition].shape[0]

    dfData = [ ["Yes", countYes], ["No", countNo] ]
    dfNACounts = pd.DataFrame(data = dfData, columns=["Result", "Count"])
    
    if conditionTitle != "":
        plt.title(conditionTitle)

    bars = plt.bar(dfNACounts["Result"], dfNACounts["Count"], label='Count')
    plt.bar_label(bars)
    return plt            


def wordCloudForTextColumn(dfInput, textColumn, maxWords=5000, imgWidth=1000, imgHeight=600):
    # Clean out empties
    dfForWordCloud = dfInput[dfInput[textColumn].str.strip() != ""]
    dfForWordCloud = dfForWordCloud[dfForWordCloud[textColumn].notna()]

    lstText = list(dfForWordCloud[textColumn])

    # Join all the texts together.
    long_string = ','.join(lstText)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=maxWords, contour_width=3, contour_color='steelblue', width=imgWidth, height=imgHeight)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()


def regressionActualVsPredictedScatter(dfTestTarget, dfTestPredicted, targetColName = ""):
    fig, ax = plt.subplots()
    ax.scatter(dfTestTarget, dfTestPredicted, s=25, cmap=plt.cm.coolwarm, zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.plot(lims, [np.mean(dfTestTarget),]*2, 'r--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if targetColName == "":
        targetColName = "Target"

    plt.xlabel('Actual ' + targetColName)
    plt.ylabel('Predicted ' + targetColName)

    plt.show()    


def regressionFeatureImportance(model, dfTrainFeatures):
    coefs = pd.DataFrame(
        model.coef_  * dfTrainFeatures.std(axis=0),
        columns=['Coefficient importance'], index=dfTrainFeatures.columns
    )
    coefs.sort_values(by=['Coefficient importance']).plot(kind='barh', figsize=(9, 7))
    plt.title('Ridge model, small regularization')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)


def matrixCorrelationAgainstTarget(df, targetCol, matrixSize):
    colNames = df.columns
    features = colNames.drop(targetCol)

    fig, axes = plt.subplots(nrows=matrixSize, ncols=matrixSize, figsize=(20, 20))
    plt.suptitle("Correlation Scatter Plots between " + targetCol + " and all other Features")

    yaxis = df[targetCol]

    featurePos = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            if featurePos >= len(features):
                break
            ax = axes[i, j]
            xaxis = df[features[featurePos]]
            ax.set_title(features[featurePos])
            ax.scatter(xaxis, yaxis)
            featurePos += 1

    plt.show()