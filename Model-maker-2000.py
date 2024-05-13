#Classifiers
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier#ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
#Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
###
from sklearn.model_selection import GridsearchCV

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


print("wlecome to model maker 2000\n\
    please enter the dataset you want to use:\n\
    1.Heart Disease Dataset\n\
    2.Cardiovascular Disease dataset\n\
    3.Heart Failure Prediction\n")

dataset_choice = (int)(input())
if dataset_choice == 1 :
    dataset_choice = "" # path for the dataset
elif dataset_choice == 2:
    dataset_choice = "" # path for the dataset
elif dataset_choice == 3:
    dataset_choice = "" # path for the dataset
##
data = pd.read_csv(dataset_choice)
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']               # Target variable
##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

please_choose_model = (int)(input("please choose the model you want to use:\n\
    Classification Models\n\
    1. Support Vector Machine\n\
    2. K-Nearest Neighbors\n\
    3. Decision Tree\n\
    4. Random Forest (ensemble)\n\
    5. Gradient Boosting Machines (ensemble)\n\
    6. Neural Network\n\
    7. Naive Bayes\n\
    8. AdaBoost\n\
    9. Linear Discriminant Analysis\n\
    10. Quadratic Discriminant Analysis\n\
    Regression Models\n\
    11.Linear Regression\n\
    12.Ridge Regression\n\
    13.Support Vector Regression\n\
    Clustering Models\n\
    14.KMeans\n\
    15.DBSCAN\n\
    16.Agglomerative Clustering\n\
    17.all models\n"))

model = [svm.SVC(), NearestNeighbors(), tree.DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier(), GaussianNB(), AdaBoostClassifier(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), LinearRegression(), Ridge(), SVR(), KMeans(), DBSCAN(), AgglomerativeClustering()]






