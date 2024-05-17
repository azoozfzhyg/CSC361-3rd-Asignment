#Classifiers
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
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
from sklearn.model_selection import GridSearchCV

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

model_choice = (int)(input("please choose the model you want to use:\n\
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

model_names = {'1':'SVM', '2':'KNN', '3':'Decision Tree', '4':'Random Forest', '5':'Gradient Boosting',
                '6':'Neural Network', '7':'Naive Bayes', '8':'AdaBoost', '9':'LDA', '10':'QDA',
                '11':'Linear Regression', '12':'Ridge Regression', '13':'SVR', '14':'KMeans', '15':'DBSCAN',
                '16':'Agglomerative Clustering'}

model_of_choice = {'SVM':svm.SVC(), 'KNN':NearestNeighbors(), 'Decision Tree':tree.DecisionTreeClassifier(),
                    'Random Forest':RandomForestClassifier(), 'Gradient Boosting':GradientBoostingClassifier(),
                    'Neural Network':MLPClassifier(), 'Naive Bayes':GaussianNB(), 'AdaBoost':AdaBoostClassifier(),
                    'LDA':LinearDiscriminantAnalysis(), 'QDA':QuadraticDiscriminantAnalysis(),
                    'Linear Regression':LinearRegression(), 'Ridge Regression':Ridge(), 'SVR':SVR(),
                    'KMeans':KMeans(), 'DBSCAN':DBSCAN(), 'Agglomerative Clustering':AgglomerativeClustering()}

model = model_of_choice[model_names[model_choice]]

#somehow get param_grids???
#Got it ;}
param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
    'Decision Tree': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (200,)], 'activation': ['relu', 'tanh', 'logistic']},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'LDA': {'solver': ['svd', 'lsqr'], 'shrinkage': [None, 'auto']},
    'QDA': {'reg_param': [0.0, 0.1, 0.2]},
    'Linear Regression': {'normalize': [True, False]},
    'Ridge Regression': {'alpha': [0.1, 1, 10]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KMeans': {'n_clusters': [2, 3, 4]},
    'DBSCAN': {'eps': [0.1, 0.5, 1.0], 'min_samples': [5, 10, 20]},
    'Agglomerative Clustering': {'n_clusters': [2, 3, 4]}
}

def hyperTuning(x_train , y_train , model):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_names[model_choice]], cv=5 , error_score='raise') # with using 5 fold cross-validation
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters for {model_names[model_choice]} : ", best_params)
    return grid_search.best_estimator_ , grid_search.best_score_

#time the model's performance
model.fit(X_train, y_train)
print(f"Results for {model_names[model_choice]} before tuning :")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
tuned_model, score = hyperTuning(X_train, y_train, model)
print(f"\nResults for {model_names[model_choice]} after tuning :")
predictions_after = tuned_model.predict(X_test)
print(classification_report(y_test, predictions_after))







