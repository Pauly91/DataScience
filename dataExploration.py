from pandas import read_csv,Series,DataFrame,to_datetime,TimeGrouper,concat,rolling_mean
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from numpy import ones,log
from pandas import to_numeric,options,tools, scatter_matrix, DataFrame
from matplotlib import pyplot
import numpy as np
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#matplotlib.pyplot.style.use = 'default'




'''

git fetch origin
git reset --hard origin/[tag/branch/commit-id usually: master]

1) Define the Problem



Step 1: What is the problem? Describe the problem informally and formally and list assumptions and similar problems.
Step 2: Why does the problem need to be solved? List your motivation for solving the problem, the benefits a solution provides and how the solution will be used.
Step 3: How would I solve the problem? Describe how the problem would be solved


2) Prepare Data

Step 1: Data Selection: Consider what data is available, what data is missing and what data can be removed.
Step 2: Data Preprocessing: Organize your selected data by formatting, cleaning and sampling from it.
Step 3: Data Transformation: Transform preprocessed data ready for machine learning by engineering features using scaling, attribute decomposition and attribute aggregation.

check for correlation in data
check for outliers in data
check for variance of feature with missing values

Spot checking for classification :

2 Linear Machine Learning Algorithms:

- Logistic Regression
- Linear Discriminant Analysis

4 Nonlinear Machine Learning Algorithms:

- K-Nearest Neighbors
- Naive Bayes
- Classification and Regression Trees
- Support Vector Machines


information value (IV)
weight of evidence (WOE)
information criteria

bi-variate analysis


outlier treatment :

capping
folding
'''
features = ['Id', 'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_4', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_4', 'Employment_Info_5', 'Employment_Info_6', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_5', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_15', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_24', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_32', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 'Medical_Keyword_1', 'Medical_Keyword_2', 'Medical_Keyword_3', 'Medical_Keyword_4', 'Medical_Keyword_5', 'Medical_Keyword_6', 'Medical_Keyword_7', 'Medical_Keyword_8', 'Medical_Keyword_9', 'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12', 'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16', 'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19', 'Medical_Keyword_20', 'Medical_Keyword_21', 'Medical_Keyword_22', 'Medical_Keyword_23', 'Medical_Keyword_24', 'Medical_Keyword_25', 'Medical_Keyword_26', 'Medical_Keyword_27', 'Medical_Keyword_28', 'Medical_Keyword_29', 'Medical_Keyword_30', 'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33', 'Medical_Keyword_34', 'Medical_Keyword_35', 'Medical_Keyword_36', 'Medical_Keyword_37', 'Medical_Keyword_38', 'Medical_Keyword_39', 'Medical_Keyword_40', 'Medical_Keyword_41', 'Medical_Keyword_42', 'Medical_Keyword_43', 'Medical_Keyword_44', 'Medical_Keyword_45', 'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48', 'Response']


continousFeatures = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']


continousFeaturesMoreOutlier = ['Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Insurance_History_5', 'Family_Hist_2', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_3']

continousFeaturesWithFullCount = ['Ins_Age', 'Ht', 'Wt', 'BMI']

def writeResults(id,result):
    # write to csv file according the format of sample_submission.csv
    # use id to save the result, make those changes

    with open('submission.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, )
        writer.writerow(['Id','Response'])
        for i in range(0,len(id)):

            writer.writerow([id[i],result[i]])



def logisticRegression(dfTrain, response,dfTest,id):
    # Work with this to create a mulitclass logistic classifier
    X = dfTrain
    Y = response
    logistic = LogisticRegression()
    logistic.fit(X, Y)
    result = logistic.predict(dfTest)
    writeResults(id,result)


def featurePreparation(features):
    print(features.describe())
    features.boxplot()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    #plt.show()

    scatter_matrix(features, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #plt.show()
    '''
    from scatter plot we that BMI and Wt are correlated

    how to choose the best features for a multi-class classification problem.

    '''
    transformedFeatures = DataFrame()
    for i in list(features.columns.values):
        transformedFeatures[i] = preprocessing.scale(boxcox(features[i] + 1)[0])

    scatter_matrix(transformedFeatures, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()
    return  transformedFeatures


def classificationSpotChecker(dfTrain, response,dfTest,id):
    logisticRegression(dfTrain, response,dfTest,id) # scored 0.24817

    '''
    logistic : scored 0.24817
    logistic + boxcox (skewness correction) : scored 0.25541

    Many model building techniques have the assumption that predictor values are distributed normally and have a symmetrical shape.
    logistic regression and normality : https://www.quora.com/Does-logistic-regression-require-independent-variables-to-be-normal-distributed

    '''

def classificationWithContVariables(dfTrain, response,dfTest,id):
    # The idea is to classification just with the continous variables

    #df.boxplot()
    #locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    #plt.show()

    dfTrain = featurePreparation(dfTrain[continousFeaturesWithFullCount])

    dfTest = featurePreparation(dfTest[continousFeaturesWithFullCount])

    classificationSpotChecker(dfTrain[continousFeaturesWithFullCount], response, dfTest[continousFeaturesWithFullCount],id)





def main():

    dfTrain = read_csv("train.csv", header=0)
    dfTest = read_csv("test.csv", header=0)
    response = dfTrain['Response']
    id = dfTest['Id']
    classificationWithContVariables(dfTrain[continousFeatures],response, dfTest[continousFeatures],id)






if __name__ == '__main__':
    main()