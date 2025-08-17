import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNN(input_data):
    diabetes_data = pd.read_csv("diabetes.csv")

    ## Independent and Dependent features
    x = diabetes_data.drop(columns = ['Outcome'], axis =1)
    y = diabetes_data['Outcome']


    ##Train Test Split

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=.2, random_state=2)

    ## Standardize the dataset
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)



    # Training the K-NN model on the Training set

    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    output = classifier.predict(scaler.transform(input_data))
    acc_score = accuracy_score(y_test, y_pred)
    return acc_score , output