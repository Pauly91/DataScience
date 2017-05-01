from pandas import read_csv,Series,DataFrame,to_datetime,TimeGrouper,concat,rolling_mean
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox,multivariate_normal
from numpy import ones,log
from pandas import to_numeric,options,tools, scatter_matrix, DataFrame
from matplotlib import pyplot
import numpy as np
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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


continousFeatures = ['Response','Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']


continousFeaturesMoreOutlier = ['Response','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Insurance_History_5', 'Family_Hist_2', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_3']

continousFeaturesWithFullCount = ['Response','Ins_Age', 'Ht', 'Wt', 'BMI']

def writeResults(id,result):
    # write to csv file according the format of sample_submission.csv
    # use id to save the result, make those changes

    with open('submission.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, )
        writer.writerow(['Id','Response'])
        for i in range(0,len(id)):

            writer.writerow([id[i],result[i]])




def logisticRegression(dfTrain ,dfTest, validation):
    # Work with this to create a mulitclass logistic classifier
    Y = dfTrain['Response']
    dfTrain = dfTrain.drop('Response', 1)
    X = dfTrain
    logistic = LogisticRegression()
    logistic.fit(X, Y)
    '''
    result = logistic.predict(dfTest)
    writeResults(id,result)

    write a routine here to test the data with the testing data set

    '''
    return logistic





def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariateGaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon


def outlierRemoval(train,test,validation):
    '''


    links on outlier detection :
    http://bugra.github.io/work/notes/2014-03-31/outlier-detection-in-time-series-signals-fft-median-filtering/
    http://bugra.github.io/work/notes/2014-05-11/robust-regression-and-outlier-detection-via-gaussian-processes/
    http://bugra.github.io/work/notes/2014-04-26/outlier-detection-markov-chain-monte-carlo-via-pymc/


How to detect outliers?

http://shahramabyari.com/2015/12/25/data-preparation-for-predictive-modeling-resolving-outliers/

There are several approaches for detecting Outliers. Charu Aggarwal in his book Outlier Analysis classifies Outlier detection models in following groups:

1. Extreme Value Analysis: This is the most basic form of outlier detection and only good for 1-dimension data. In these types of analysis, it is assumed that values which are too large or too small are outliers. Z-test and Student’s t-test are examples of these statistical methods. These are good heuristics for initial analysis of data but they don’t have much value in multivariate settings. They can be used as final steps for interpreting outputs of other outlier detection methods.

2. Probabilistic and Statistical Models: These models assume specific distributions for data. Then using the expectation-maximization(EM) methods they estimate the parameters of the model. Finally, they calculate probability of membership of each data point to calculated distribution. The points with low probability of membership are marked as outliers.

3. Linear Models: These methods model the data into a lower dimensional sub-spaces with the use of linear correlations. Then the distance of each data point to plane that fits the sub-space is being calculated. This distance is used to find outliers. PCA(Principal Component Analysis) is an example of linear models for anomaly detection.

4. Proximity-based Models: The idea with these methods is to model outliers as points which are isolated from rest of observations. Cluster analysis, density based analysis and nearest neighborhood are main approaches of this kind.

5. Information Theoretic Models: The idea of these methods is the fact that outliers increase the minimum code length to describe a data set.

6. High-Dimensional Outlier Detection: Specifc methods to handle high dimensional sparse data


    Next, define a function to find the optimal value for threshold (epsilon) that can be used to differentiate between normal and anomalous data points. For learning the optimal value of epsilon we will try different values in a range of learned probabilities on a cross-validation set. The f-score will be calculated for predicted anomalies based on the ground truth data available. The epsilon value with highest f-score will be selected as threshold i.e. the probabilities that lie below the selected threshold will be considered anomalous.

This seems to be the best article on multivariate data :

https://aqibsaeed.github.io/2016-07-17-anomaly-detection/

read and understand the algorithm


    '''
    responseTrain = train['Response']
    train = train.drop('Response', 1)

    responseValidation = validation['Response']
    print(responseValidation)
    validation = validation.drop('Response', 1)

    mu, sigma = estimateGaussian(train)
    p = multivariateGaussian(train, mu, sigma)
    print(p)
    p_cv = multivariateGaussian(validation, mu, sigma)
    fscore, ep = selectThresholdByCV(p_cv, responseValidation)
    outliers = np.asarray(np.where(p < ep))

    print(outliers)
    print(len(outliers[0]))
    '''
    Indexed of outliers are produced here :

    Next Steps :

    1. build validation dataset
    2. learn more about the algo
    3. refer the website : https://aqibsaeed.github.io/2016-07-17-anomaly-detection/

    Lesson :

    outlier removal is done on the entire dataset, ie considering each point as mulitdimension and
    not feature by feature, but what about categorical variable is the question.
    '''

def featurePreparation(features):
    responseDf = DataFrame()
    responseDf['Response']  = features['Response']

    '''
    issue in adding and dropping a feature

    originally int64
    afterwards its float64

    '''
    features = features.drop('Response', 1)
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



    #scatter_matrix(transformedFeatures, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #plt.show()

    transformedFeatures['Response'] = responseDf['Response']
    print(responseDf['Response'] )
    print(transformedFeatures['Response'])
    return  transformedFeatures


def classificationSpotChecker(dfTrain, dfTest, validation):
    model = logisticRegression(dfTrain, dfTest, validation) # scored 0.24817
    return model
    '''
    logistic : scored 0.24817
    logistic + boxcox (skewness correction) : scored 0.25541

    Many model building techniques have the assumption that predictor values are distributed normally and have a symmetrical shape.
    logistic regression and normality : https://www.quora.com/Does-logistic-regression-require-independent-variables-to-be-normal-distributed

    '''

def classificationWithContVariables(dfTrain, dfTest, validation):
    # The idea is to classification just with the continous variables

    #df.boxplot()
    #locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    #plt.show()

    dfTrain = featurePreparation(dfTrain[continousFeaturesWithFullCount])

    dfTest = featurePreparation(dfTest[continousFeaturesWithFullCount])

    validation = featurePreparation(validation[continousFeaturesWithFullCount])



    outlierRemoval(dfTrain,dfTest,validation)
    model = 0
    #model = classificationSpotChecker(dfTrain[continousFeaturesWithFullCount], response, dfTest[continousFeaturesWithFullCount],id, validation[continousFeaturesWithFullCount])
    return model

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test



def main():

    dfTrain = read_csv("train.csv", header=0)
    dfTest = read_csv("test.csv", header=0)


    train, validate, test = train_validate_test_split(dfTrain)

    model = classificationWithContVariables(train[continousFeatures],test[continousFeatures], validate[continousFeatures])

    id = dfTest['Id']
    result = model.predict(dfTest)
    writeResults(id, result)




if __name__ == '__main__':
    main()