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
from sklearn.cross_validation import StratifiedKFold
matplotlib.style.use('ggplot')





























def dataSplitter(df,y,type):
    '''
    
    :param df: The data  
    :param y: responses for data
    :param type: type of ML problem
    :return: Train and validation data
    
    
    Read about Stratified splitting of data.
    
    Material on class imbalance : https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
    
    '''
    ySplit = y.values
    print(np.shape(ySplit))
    ySplit = np.reshape(ySplit,[len(ySplit),])
    if type == 1: # Imbalanced classification

        eval_size = 0.10
        kf = StratifiedKFold(ySplit[:], round(1./eval_size),shuffle=True)
        train_indices, valid_indices = next(iter(kf))
        print(max(train_indices))
        print(max(valid_indices))
        x_train, y_train= df.ix[train_indices], y.ix[train_indices]
        x_valid, y_valid = df.ix[valid_indices], y.ix[valid_indices]

    # fill rest of the methods for splitting the data

    return  x_train, y_train, x_valid, y_valid








# Trying to implement this : http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/


def main():



    dfTrain = read_csv("train.csv", header=0)
    dfTest = read_csv("test.csv", header=0)

    y = DataFrame(dfTrain['Response'].astype('category'))
    dfTrain.drop('Response',axis=1,inplace=True)
    dfTrain.drop('Id', axis=1,inplace=True)

    yCount = y['Response'].value_counts()
    print(yCount)
    '''
    yCount = y['Response'].value_counts()
    print(yCount)
    
    The above results reveal that it is imbalanced multi-class classification problem 
    
    # Use of Stratified splitting; ref : http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/

    '''
    [x_train, y_train, x_valid, y_valid] = dataSplitter(dfTrain, y, 1)



    yCount = y_train['Response'].value_counts()
    print(yCount)
    yCount = y_valid['Response'].value_counts()
    print(yCount)





if __name__ == '__main__':
    main()


    '''
    
    Data fields

Variable	Description
Id	A unique identifier associated with an application.
Product_Info_1-7	A set of normalized variables relating to the product applied for
Ins_Age	Normalized age of applicant
Ht	Normalized height of applicant
Wt	Normalized weight of applicant
BMI	Normalized BMI of applicant
Employment_Info_1-6	A set of normalized variables relating to the employment history of the applicant.
InsuredInfo_1-6	A set of normalized variables providing information about the applicant.
Insurance_History_1-9	A set of normalized variables relating to the insurance history of the applicant.
Family_Hist_1-5	A set of normalized variables relating to the family history of the applicant.
Medical_History_1-41	A set of normalized variables relating to the medical history of the applicant.
Medical_Keyword_1-48	A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.
Response	This is the target variable, an ordinal variable relating to the final decision associated with an application
The following variables are all categorical (nominal):

Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41

The following variables are continuous:

Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

The following variables are discrete:

Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

Medical_Keyword_1-48 are dummy variables.
    '''