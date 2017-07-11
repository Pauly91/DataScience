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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

matplotlib.style.use('ggplot')


catVar = ['Product_Info_1','Product_Info_2','Product_Info_3','Product_Info_5','Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16','Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']

contVar = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2, Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

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




def categoricalDataAnalysis(df):
    dtype_df = df[catVar].dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]

    print(df['Product_Info_2'].value_counts()) # The only categorical data without numerical val



def categoricalDataHandling(df):

    # Converts labels to numbers i.e encodes the data
    labelEncoder = LabelEncoder()
    labelEncoder.fit(df['Product_Info_2'])
    product_Info_2LabelEncoder = labelEncoder.transform(df['Product_Info_2'])
    df['product_Info_2LabelEncoded'] = product_Info_2LabelEncoder

    df.drop('Product_Info_2', axis = 1, inplace = True)
    print(df.head(10))


    # Refer this : http://biggyani.blogspot.in/2014/08/using-onehot-with-categorical.html
    # The idea to transform DataFrame to np.array and process it. Read about it more to get a feel of it


    train_categorical_values = np.array(df)
    ohe = OneHotEncoder()

    train_cat_data = ohe.fit_transform(train_categorical_values)

    train_cat_data_df = DataFrame(train_cat_data.toarray())
    print(train_cat_data_df.describe)



    # PCA on the data

    pca = PCA(n_components=60) # how to choose the components

    train_cat_data_array = np.array(train_cat_data_df)
    pca.fit(train_cat_data_df)
    train_cat_data_PCA = pca.transform(train_cat_data_array)
    print(len(train_cat_data_PCA))
    train_cat_data_df_pca = DataFrame(train_cat_data_PCA)
    print(train_cat_data_df_pca.describe)

    return train_cat_data_df_pca


def getBestFeatures(df,y):

    # read this :https://jessesw.com/XG-Boost/

    model =


def classificationSpotChecker(df,y):

    dfBestt = getBestFeatures(df,y)




# Trying to implement this :
# http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/


def main():



    dfTrain = read_csv("train.csv", header=0)
    dfTest = read_csv("test.csv", header=0)

    y = DataFrame(dfTrain['Response'].astype('category'))
    dfTrain.drop('Response',axis=1,inplace=True)
    dfTrain.drop('Id', axis=1,inplace=True)

    '''
    yCount = y['Response'].value_counts()
    print(yCount)
    
    The above results reveal that it is imbalanced multi-class classification problem 
    
    # Use of Stratified splitting; ref : http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/

    '''
    [x_train, y_train, x_valid, y_valid] = dataSplitter(dfTrain, y, 1)
    print(x_train[catVar].head(10))
    categoricalDataAnalysis(x_train[catVar])
    df = categoricalDataHandling(x_train[catVar])

    classificationSpotChecker(df,y_train)






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