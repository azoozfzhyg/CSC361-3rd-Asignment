#Classifiers
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier #ensemble
from sklearn.ensemble import GradientBoostingClassifier #ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#Regressors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, SVR
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# data1 = pd.read_csv("C:\\Users\\Admin\\Desktop\\DataSets\\heart_data.csv")
# data2 = pd.read_csv("C:\\Users\\Admin\\Desktop\\DataSets\\cardio_train.csv")
# data3 = pd.read_csv("C:\\Users\\Admin\\Desktop\\DataSets\\heart_failure_clinical_records_dataset.csv")

def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    report = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² Score': r2
    }

print("wlecome to model maker 2000\n\
    please enter the dataset you want to use:\n\
    1.Heart Disease Dataset\n\
    2.Cardiovascular Disease dataset\n\
    3.Heart Failure Prediction\n")

dataset_choice = (int)(input())
if dataset_choice == 1 :
    dataset_choice = "heart_data.csv" # path for the dataset
elif dataset_choice == 2:
    dataset_choice = "cardio_train.csv" # path for the dataset
elif dataset_choice == 3:
    dataset_choice = "heart_failure_clinical_records_dataset.csv" # path for the dataset
##

data = pd.read_csv(dataset_choice)
X = data.drop(data.columns[-1], axis=1) 
y = data[data.columns[-1]]  # Target variable
X = StandardScaler().fit_transform(X)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#somehow get param_grids???
#Got it ;}

model_names = {'1':'SVM', '2':'KNN', '3':'Decision Tree', '4':'Random Forest', '5':'Gradient Boosting',
            '6':'Neural Network', '7':'GaussianNB', '8':'Voting Classifier', '9':'LDA', '10':'QDA',
            '11':'Linear Regression', '12':'Ridge Regression', '13':'SVR', '14':'KMeans', '15':'DBSCAN',
            '16':'Agglomerative Clustering'}

model_of_choice = {'SVM':svm.SVC(), 'KNN':NearestNeighbors(), 'Decision Tree':tree.DecisionTreeClassifier(),
                'Random Forest':RandomForestClassifier(), 'Gradient Boosting':GradientBoostingClassifier(),
                'Neural Network':MLPClassifier(), 'GaussianNB':GaussianNB(), 'Voting Classifier':VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)), ('svc', SVC(probability=True)), ('rf', RandomForestClassifier())]),
                'LDA':LinearDiscriminantAnalysis(), 'QDA':QuadraticDiscriminantAnalysis(),
                'Linear Regression':LinearRegression(), 'Ridge Regression':Ridge(), 'SVR':SVR(),
                'KMeans':KMeans(), 'DBSCAN':DBSCAN(), 'Agglomerative Clustering':AgglomerativeClustering()
}

param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
    'Decision Tree': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (200,)], 'activation': ['relu', 'tanh', 'logistic'], 'max_iter': [5000], 'learning_rate_init': [0.0001, 0.001, 0.01]},
    'GaussianNB': {'var_smoothing': [1e-09, 1e-08, 1e-07]},
    'Voting Classifier': {'voting': ['hard', 'soft'], 'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]},
    'LDA': {'solver': [ 'lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 1.0], 'n_components': [None, 1], 'tol': [1e-4, 1e-3, 1e-2]}, # does not work
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

model_choice = (input("please choose the model you want to use:\n\
Classification Models\n\
1. Support Vector Machine\n\
2. K-Nearest Neighbors\n\
3. Decision Tree\n\
4. Random Forest (ensemble)\n\
5. Gradient Boosting Machines (ensemble)\n\
6. Neural Network\n\
7. GaussianNB\n\
8. Voting Classifier\n\
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
17.all models\n\
18.Exit\n"))

while model_choice != '18':  # This creates an infinite loop

    model = model_of_choice[model_names[model_choice]]
    
    #time the model's performance
    model.fit(X_train, y_train)
    print(f"Results for {model_names[model_choice]} before tuning :")
    predictions = model.predict(X_test)
    if model_choice in range(1, 10):
        print(classification_report(y_test, predictions))
    elif model_choice in range(11, 13):
        print(regression_report(y_test, predictions))
    else:
        pass # clustering models report not implemented yet
    tuned_model, score = hyperTuning(X_train, y_train, model)
    print(f"\nResults for {model_names[model_choice]} after tuning :")
    predictions_after = tuned_model.predict(X_test)
    print(classification_report(y_test, predictions_after))
    
    model_choice = (input("please choose the model you want to use:\n\
    Classification Models\n\
    1. Support Vector Machine\n\
    2. K-Nearest Neighbors\n\
    3. Decision Tree\n\
    4. Random Forest (ensemble)\n\
    5. Gradient Boosting Machines (ensemble)\n\
    6. Neural Network\n\
    7. GaussianNB\n\
    8. Voting Classifier\n\
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
    17.all models\n\
    18.Exit\n"))

