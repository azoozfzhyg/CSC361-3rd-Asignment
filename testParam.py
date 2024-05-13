import sklearn
from sklearn.model_selection import GridSearchCV

def hyperTuning(x_train , y_train , modelName):  # Example of hyperparameter tuning for RandomForest
    grid_search = GridSearchCV(estimator=classification_models[modelName], param_grid=param_grids[modelName], cv=5 , error_score='raise') # with using 5 fold cross-validation
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters for {modelName} : ", best_params)
    return grid_search.best_estimator_ , grid_search.best_score_