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
