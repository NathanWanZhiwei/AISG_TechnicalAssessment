# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:24:30 2022
Updated on Tues April 12 10:59 2022

@author: Nathan Wan Zhiwei
"""

# Importing Self Defined Python Packages
from Packages import EDAFunctions as eda

# Importing Python Packages for Machine Learning
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, recall_score, make_scorer
from sklearn.model_selection import cross_val_score
import configparser

"""
Perform data clean up on the extracted dataset.

Previously, all the data cleanup codes are placed here. But now,
the codes are residing in EDAFunctions package. This makes the code
cleaner and easier to read.

Learned Functions Being Applied Within EDAFunctions
***************************************************
1) Apply map function to standardizing values within a dataframe
2) Defined function for repetitive tasks
3) Apply appropriate docstrings commenting
4) Apply the usage of lambda
5) Error Exceptions for functions
6) Creating Python Packages
7) Utilizes inner function
"""

# Load dataset
surviveDataset = eda.ExtractDataset("../data/survive.db")

# Remove NULL Values from dataset
surviveDataset = eda.DropNullValues(surviveDataset)

# Remove "favorite color" from Dataset
surviveDataset = eda.DropDataColumn(surviveDataset, 'Favorite color')

#Standardize "Smoke" and "Survive" values to 1 and 0
surviveDataset = eda.StandardizeYesNoColumn(surviveDataset,'Smoke')
surviveDataset = eda.StandardizeYesNoColumn(surviveDataset,'Survive')

# Standardizing "Ejection Fraction" Values
surviveDataset = eda.StandardizeEjectionFractionColumn(surviveDataset,'Ejection Fraction')

# Removing Negative Age Values
surviveDataset = eda.RemoveNegativeValuesFromColumn(surviveDataset, 'Age')

# Change "Diabetes" for Normal to "0" and "Pre-Diabetes" to "1" and "Diabetes" to "2"
surviveDataset = eda.StandardizingDiabetesColumn(surviveDataset, 'Diabetes')

#Change "Male" to "1" and "Female" to "0"
surviveDataset = eda.StandardizingGenderColumn(surviveDataset, 'Gender')

# Drop ID for the machine learning pipeline. It does not contribute to the prediction.
surviveDataset = eda.DropDataColumn(surviveDataset, 'ID')

"""
From here on will be the machine learning functions. Since we have demonstrated the use of
using python packages in the previous EDA section, all machine learning functions will be
written here to demonstrate knowledge learnt for the Machine Learning Track

Learned Functions Being Applied Within Machine Learning
*******************************************************
1) Log normalization on data with high varience
2) Updated model usage using classification models
3) Cross validation 
4) Measuring Model Accuracy (Confusion Matrix?)
"""
# Before performing log normalization, 'Pletelets' and 'Creatinine phosphokinase' have a very high varience
print( "Pletelets Varience Before Log Normalization: " + str( surviveDataset['Pletelets'].var( ) ) )#Reports a varience of 9623618754.77551
surviveDataset['Pletelets'] = np.log(surviveDataset['Pletelets'])  
print( "Pletelets Varience After Log Normalization: " + str( surviveDataset['Pletelets'].var( ) ) )#Reports a varience of 0.1596048918991591

print("Creatinine phosphokinase Varience Before Log Normalization: " + str( surviveDataset['Creatinine phosphokinase'].var( ) ) )#Reports a varience of 919883.6200256263
surviveDataset['Creatinine phosphokinase'] = np.log(surviveDataset['Creatinine phosphokinase'])
print("Creatinine phosphokinase Varience After Log Normalization: " + str( surviveDataset['Creatinine phosphokinase'].var( ) ) + "\n" )#Reports a varience of 1.2729947744163095
#After performing log normalization, 'Pletelets' and 'Creatinine phosphokinase' varience has now decreased drastically

"""
In the previous attempt, some models used were "LogisticRegression" or "GaussianNB" etc. However, 
since this model is attempting to solve a classification problem (survive or dont survive), we will update
the model being used. We shall use the 2 models covered within the course that addresses classication 
problems: DecisionTreeClassifier and KNeighborsClassifier
"""
dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()

# Lets Create a Training Dataset and Validation Dataset
X = surviveDataset.drop( columns=[ 'Survive' ] )
y = surviveDataset[ 'Survive' ]

"""
Here, we will also demonstrate the usage of cross validation using sklearn's cross_val_score()
"""
mse = make_scorer(mean_absolute_error)
cv_dtc = cross_val_score(estimator=dtc, 
                         X=X,
                         y=y,
                         cv=10,
                         scoring=mse)

cv_knn = cross_val_score(estimator=knn, 
                         X=X,
                         y=y,
                         cv=10,
                         scoring=mse)

print("cross_val_score DTC: " + str(cv_dtc.mean()))
print("cross_val_score KNN: " + str(cv_knn.mean()))

"""
Here, we will also demonstrate the usage of cross validation. Although there is a cross_val_score() function, we will
use a for loop to demonstrate the understanding of cross validation.

We will also focus on the recall_score of the training model as we cannot afford to miss any values where the patient survives.
Accuracy and precision would not be used here.
"""

# Remove the header row
X = X.values
y = y.values

kf = KFold(n_splits=10,random_state=1111,shuffle=True);
splits = kf.split(X)

dtc_error = []
dtc_recall = []
knn_error = []
knn_recall = []

for train_index, val_index in splits:
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index] 
    
    #Fit the training set into the decision tree classifier
    dtc.fit(X_train, y_train)
    #Test with validation dataset
    dtc_predictions = dtc.predict(X_val)
    #Append the mean squared error and recall error result to a list
    dtc_error.append(mean_absolute_error(y_val,dtc_predictions))
    dtc_recall.append(recall_score(y_val,dtc_predictions))
    
    #Fit the training set into the K-Nearest Neighbours classifier
    knn.fit(X_train, y_train)
    #Test with validation dataset
    knn_predictions = knn.predict(X_val)
    #Append the mean squared error and recall error result to a list
    knn_error.append(mean_absolute_error(y_val,knn_predictions))
    knn_recall.append(recall_score(y_val,knn_predictions))
    
#Print the average of each classifier's mean squared error
print("Mean Decision Tree Classifier Error: " + str(np.mean(dtc_error)))
print("Mean Decision Tree Classifier Recall Error: " + str(np.mean(dtc_recall)))
print("Mean K-Nearest Neighbour Classifier Error: " + str(np.mean(knn_error)))
print("Mean K-Nearest Neighbour Classifier Recall Error: " + str(np.mean(knn_recall)) + "\n")


"""
It is noticed that using the for method to perform cross validation yields the same results
as cross_val_score() function.
"""

# Read from config files the values for patient dataset
parser = configparser.ConfigParser( )
parser.read( "Config.ini" )
gender=parser.get( "Patient", "Gender" ) 
smoke=parser.get( "Patient", "Smoke" ) 
diabetes=parser.get( "Patient", "Diabetes" ) 
age=parser.get( "Patient", "Age" ) 
ejectionFraction=parser.get( "Patient", "EjectionFraction" ) 
sodium=parser.get( "Patient", "Sodium" ) 
creatinine=parser.get( "Patient", "Creatinine" ) 
pletelets=parser.get( "Patient", "Pletelets" ) 
creatininePhosphokinase=parser.get( "Patient", "CreatininePhosphokinase" ) 
bloodPressure=parser.get( "Patient", "BloodPressure" ) 
hemoglobin=parser.get( "Patient", "Hemoglobin" ) 
height=parser.get( "Patient", "Height" ) 
weight=parser.get( "Patient", "Weight" ) 

# Print Out Dataset Details
print( "Patient Details: " )
print( "Gender :" , gender )
print( "Smoke :" , smoke )
print( "Diabetes :" , diabetes )
print( "Age :" , age )
print( "Ejection Fraction :" , ejectionFraction )
print( "Sodium :" , sodium )
print( "Creatinine :" , creatinine )
print( "Pletelets :" , pletelets )
print( "Creatinine Phosphokinase :" , creatininePhosphokinase )
print( "Blood Pressure :" , bloodPressure )
print( "Hemoglobin :" , hemoglobin )
print( "Height :" , height )
print( "Weight :" , weight , "\n" )

patientData = { 'Gender': [ gender ],
                'Smoke': [ smoke ],
                'Diabetes': [ diabetes ],
                'Age': [ age ],
                'Ejection Fraction': [ ejectionFraction ],
                'Sodium': [ sodium ],
                'Creatinine': [ creatinine ],
                'Pletelets': [ pletelets ],
                'Creatinine phosphokinase': [ creatininePhosphokinase ],
                'Blood Pressure': [ bloodPressure ],
                'Hemoglobin': [ hemoglobin ],
                'Height': [ height ],
                'Weight': [ weight ] }

patientDataFrame = pd.DataFrame( patientData )

# Insert the data to predict
predictions = knn.predict( patientDataFrame )

print( "Prediction Value: " ,predictions )

# Print out the prediction
if predictions == 1:
    print( "Patient Survives\n" )
else :
    print( "Patient Dies\n" )    






