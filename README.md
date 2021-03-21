# machine-learning_-analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score #sklearn
 
from sklearn import svm #import svm tech.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc#for accuracy score
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import metrics
import seaborn as sn

# data=pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P2_Data.csv")
data=pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P2_Data.csv")
#load the excel file via pandas

data

data['F15'].fillna((data['F15'].mean()), inplace=True)
#mean of f15 column which consist NaN values

data.isna().sum()

data['Class'].value_counts() #count the values

from sklearn.preprocessing import LabelEncoder # Label_Encoder from sklearn for string values in class column to convert in 0 and 1. 
LE = LabelEncoder()
data['Class'] = LE.fit_transform(data['Class'])


from sklearn.preprocessing import MinMaxScaler # scaler for scale the data between 0 and 1.
scaler = MinMaxScaler(feature_range=[0,1])

from sklearn.preprocessing import MinMaxScaler # scaler for scale the data between 0 and 1.
scaler = MinMaxScaler(feature_range=[0,1])

sn.countplot(x='Class',data=data) # plot the data

x = data.drop(['Class'], axis=1) # define and drop the class column 
y = data['Class'] # define y column

from sklearn.model_selection import train_test_split
# import the train_test_split from sklearn  

x = scaler.fit_transform(x)
#Standardize features by removing the mean and scaling to unit variance

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3)
# split the training between 70% and 30% ratio.







from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train) #fit data x_train and y_train
#import decesion tree from sklearn and fit into train data (decision tree classifier), fit into x_train and y_train     

y_pred = classifier.predict(x_test) #prediction on x_test
y_pred #prediction function  

 y_pred.size #size of prediction

<span style='color:red;font-weight:bold;font-size:20px'>Here define data_tes.</span>

data_tes=pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P2_Test.csv") # test data defines

data_tes 

data_tes['F15'].fillna((data_tes['F15'].mean()), inplace=True) # mean for blank data in test dataset

x_test1 = scaler.transform(data_tes.iloc[:,:-1])# scaling the dataset

y_pred = classifier.predict(x_test1) # perform prediction

#decision tree predictions
y_pred

# perform_kNN_classifier

import random
random.seed(123)
accuracies = {}
scoreList = []
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train, y_train.values.ravel())
    scoreList.append(knn2.score(x_test, y_test.values.ravel()))
    
plt.plot(range(1,15), scoreList)
plt.xticks(np.arange(1,15,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))

# maximum knn score is 79.56% 
# So getting the best neighbors number which gave you the max score
# you may use this neighbors as this gave you the max score in above experiment
neighbors = np.argmax(scoreList)
neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)  # n_neighbors means k
knn.fit(x_train, y_train.values.ravel())
prediction_knn = knn.predict(x_test)

print("{} NN Score: {:.2f}%".format(5, knn.score(x_test, y_test)*100))

prediction_knn # prediction of kNN by declaring prediction_knn

data=pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P2_Test.csv")

data['F15'].fillna((data['F15'].mean()), inplace=True) #find the mean of data 

x_test_knn_predictions = data.iloc[:,0:15]

x_test_knn_predictions

data['Class'] = knn.predict(x_test_knn_predictions)

knn_predictions = knn.predict(x_test1) # getting prediction of via knn_prediction

knn_predictions.shape # shape of prediction of knn data

# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE1 =LabelEncoder()
data['Class'] = LE.fit_transform(data['Class'])


sn.countplot(x='Class',data=data) #plotting

data['F15'].fillna((data['F15'].mean()), inplace=True) # mean for f15 column

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.3, random_state=20) # splitting train and test data by the size of 70% and 30%.

from sklearn.preprocessing import MinMaxScaler # scaling the data
scaler = MinMaxScaler(feature_range=[0,1]) # scaling the data between 0 and 1
scaler_y = MinMaxScaler()

x = data.drop(['Class'], axis=1) # dropping the class
y = data['Class'] # define the y

from sklearn.svm import SVC # svm classifier fitting
svclassifier = SVC(kernel='linear') #used linear kernel 
svclassifier.fit(x_train, y_train) # fitting x_train and y_train

y_pred = svclassifier.predict(x_test) # using svc classifier prediction of x_test

y_pred # get a prediction of y_pred

<p style='color:red;font-size:20px'>Compare the scores of all the classifiers and use the one which gives you highest score on your test/val dataset.
<br/><br/>
Like following.</p>

# DecisionTree Classifier Score
classifier.score(x_test, y_test)

# Score of KNN classifier
knn.score(x_test,y_test)

# score of SVC classifier
svclassifier.score(x_test,y_test)

final_predictions = svclassifier.predict(x_test1) # final predictions

data_tes['Class'] = final_predictions
data_tes['Class'].replace({0:'False', 1:'True'}, inplace=True)

data_tes.to_csv('/Users/nirmalpatel/Downloads/knn_new_test_off.csv', index=False)
# by using .to_csv function, without adding index adding the values in class after that successfuly converted into csv file.



# Prediction on Test Dataset.


# importing numpy,sklearn, pandas , svm technique, knn technique, decision tree regression , matplotlib , metrics in order to define mean squared error and train and test split.
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sn
import matplotlib.pyplot as plt

# loading data from excel file via data and data_test using pd.read_csv.
data = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Data.csv")
data_test = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test.csv")

data.head() # using data.head() we got first 5 rows of data.



# counting NaN values from columns
data.isna().sum()

# Standardize features by removing the mean and scaling to unit variance
# declared two function for encoder because there are two columns
# f15 and f8 have string values 
#label encoder could be imported from sklearn
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE1 =LabelEncoder()
data['F15'] = LE.fit_transform(data['F15'])
data['F8'] = LE1.fit_transform(data['F8'])
data_test['F15'] = LE.transform(data_test['F15'])
data_test['F8'] = LabelEncoder().fit_transform(data['F8'])

# filling missing values  


data['F8'].fillna(0, inplace=True)

data['F15'].fillna(0, inplace=True)

data_test['F8'].fillna(0, inplace=True)

data_test['F15'].fillna(0, inplace=True)

# counts empty values from target
data['Target'].value_counts()

# scaling the data from sklean.preprocessing using importing MinMaxScaler by ranging them between 0 and 1. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0,1])
scaler_y = MinMaxScaler()

 

# integer-location based indexing by position.
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x.shape # getting a shape of x.

# splitting training and testing data with the ratio of 80% training data and 20% of testing data.
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.2, random_state=20)

 

 

# 
x_train = scaler.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))

x_test = scaler.transform(x_test)
y_test = scaler_y.transform(y_test.values.reshape(-1,1))



data_test.head()

# .ravel() returns a view of the original array whenever possible.
# using support vector implementing svm model and fit the model, this time it is svm regression model as we have to perform regression on P3 train data set. 
SupportVectorRegModel = SVR()
SupportVectorRegModel.fit(x_train,y_train.ravel())

# via svm regression model , predicted x_train model 
SupportVectorRegModel.predict(x_train) 

# i used .score to get a score of x_test and y_test.
SupportVectorRegModel.score(x_test,y_test)

# score of svm is 85%.

data_test.head() # first five column of data_test

# x_test1 for scaling data_test using iloc
x_test1 = scaler.transform(data_test.iloc[:,:-1])

#y_pred define using support regression model ,by that we will get predictions.
y_pred = SupportVectorRegModel.predict(x_test1)


#inverse the data so predictions function could predict from real data instead of 0 and 1.
predictions = scaler_y.inverse_transform(y_pred.reshape(-1,1))

predictions

# insert all the values in target column from predictions.
data_test['Target'] = predictions

# first five values from data_test.
data_test.head()

# converting data_test into csv using .to_csv function and index=false so it won't include index of 0 to 10.
# all the data will be saved in excel file in target column.
data_test.to_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test_res.csv", index=False)

 

#knn

# load the test data.
data = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test.csv")

data

# counting null values or NaN values from columns.    
data.isna().sum()

data.isna().sum()

# by using y_pred, insert all the values in target column.
data['Target']= y_pred

# get a size of data.
data.size()

# print a first five rows from data.
data.head()

data.index



#getting a shape of y_pred
y_pred.shape







# create a function name prediction_knn_new.
prediction_knn_new = pd.DataFrame(knn_predictions)

prediction_knn_new



#Linear_Regression

# load data from excel file via pandas using variable name data and data_test.
# working simultaneoulsy with both files including data_test and data as well.
data = pd.read_csv('/Users/nirmalpatel/Downloads/CE802_P3_Data.csv') 
data_test = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test.csv")

# Load the data.
data

# count NaN values from columns and rows.
data.isna().sum()

# import labelencoder from sklearn so that string values can be converted into format that notebook can read.
# defined two encoder so that each encoder have its own function. 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE1 =LabelEncoder()
data['F15'] = LE.fit_transform(data['F15'])
data['F8'] = LE1.fit_transform(data['F8'])
data_test['F15'] = LE.transform(data_test['F15'])
data_test['F8'] = LabelEncoder().fit_transform(data['F8'])

# fill the NaN values.
data['F8'].fillna(0, inplace=True)

data['F15'].fillna(0, inplace=True)

data_test['F8'].fillna(0, inplace=True)

data_test['F15'].fillna(0, inplace=True)

data.isna().sum()

# count blank values from target.
data['Target'].value_counts()

# scaling the dataset in 0 to 1 so process could be easy.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0,1])
scaler_y = MinMaxScaler() 

# dropping the target column 
x = data.drop(['Target'], axis=1)
y = data['Target']

# scaling and fit_transform.
x = scaler.fit_transform(x)
y = scaler_y.fit_transform(y.values.reshape(-1,1))

# split train and test data by size of 70% and 30%.
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=20)

# Implementing LinearRegression model from sklearn.
#fitting model on data set.
#create y_pred after fitting the data.
from sklearn.linear_model import LinearRegression
regressior = LinearRegression()
regressior.fit(x_train,y_train) 
y_pred = regressior.predict(x_train) 

# prediction via the y_pred
y_pred

# getting the regression score which is 63%.
regressior.score(x_test,y_test)

#getting data_test and used head because we can get first four rows.
data_test.head()

data.isna().sum()

# defining x_test1
x_test1 = scaler.transform(data_test.iloc[:,:-1])

# define y_pred.
y_pred = regressior.predict(x_test1)

y_pred

# f15 column and take a mean so that we can deal with blank values.

data['F15'].fillna((data['F15'].mean()), inplace=True)

# f15 column and take a mean so that we can deal with blank values.
# f8 column and take a mean so that we can deal with blank values.
data['F8'].fillna((data['F8'].mean()), inplace=True)
data['Target'].fillna((data['Target'].mean()), inplace=True)

# define testing
testing = data.drop(['F15','F9','Target'], axis=1)

linear_testing = y_pred = regressior.predict(x_test1)

# loading data_test
data_test = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test.csv")

# defining y_pred to predict and reshaping it to avoid index error.
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))

# after getting successfully prediction from data, we insert all the data in target column.
data_test['Target'] = y_pred

data_test.head(30) # getting first 30 rows form data_test dataset.

# convert data_test to csv using .to_csv and saves in excel file.
data_test.to_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test_linres.csv")






#knn_regression

# load data from excel file using data and data_test function.

data = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Data.csv")
data_test = pd.read_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test.csv")

# import labelencoder from sklearn so that string values can be converted into format that notebook can read.
# defined two encoder so that each encoder have its own function. 


# import labelencoder from sklearn so that string values can be converted into format that notebook can read.
# defined two encoder so that each encoder have its own function. 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE1 =LabelEncoder()
data['F15'] = LE.fit_transform(data['F15'])
data['F8'] = LE1.fit_transform(data['F8'])
data_test['F15'] = LE.transform(data_test['F15'])
data_test['F8'] = LabelEncoder().fit_transform(data['F8'])

# import train and test data from sklearn.
# split data based on 70% and 30%.

from sklearn.model_selection import train_test_split
train , test = train_test_split(data, test_size = 0.3)

x_train = train.drop('Target', axis=1)
y_train = train['Target']

x_test = test.drop('Target', axis = 1)
y_test = test['Target']

# scale the data using minmaxscaler to 0 and 1 to avoid error.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

#import neighbors from sklearn for kNN regression
#import sqrt function from math for mean squared error
#import matplotlib
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt

# applying kNN regression on dataset.
#calculate rmse values.
rmse_val = [] 
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  
    pred=model.predict(x_test) 
    error = sqrt(mean_squared_error(y_test,pred)) 
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

curve = pd.DataFrame(rmse_val)  
curve.plot()

# get first few rows from data_test
data_test.head()

# define x_test1
x_test1 = scaler.transform(data_test.iloc[:,:-1]) 

# define y_pred for prediciton
y_pred = model.predict(x_test1)
 


y_pred

# cleaning the data_test
data['F8'].fillna((data['F8'].mean()), inplace=True)
data['Target'].fillna((data['Target'].mean()), inplace=True)

# define testing 
testing = data.drop(['F15','F9','Target'], axis=1)

# finally knn_testing using x_test1
knn_testing = y_pred = model.predict(x_test1)

# create y_pred and use inverse function and reshape it so that we can predict from original data set and also avoid the index error in dataset.
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))

# insert all the values in the target column by y_pred
data_test['Target'] = y_pred

# import data from test prediction to target column. convert into csv (excel file).
data_test.to_csv("/Users/nirmalpatel/Downloads/CE802_P3_Test_knnres.csv")

