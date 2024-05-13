from sklearn import GridSearchCV, svm
from sklearn.neighbors import KDTree



for name, model in classification_models.items():
        model.fit(x_train, y_train)
        print(f"Results for {name} before tuning :")
        predictions = model.predict(x_test)
        print(classification_report(y_test, predictions))
        tuned_model, score =hyperTuning(x_train, y_train, name)
        print(f"\nResults for {name} after tuning :")
        predictions_after = tuned_model.predict(x_test)
        print(classification_report(y_test, predictions_after))
        if score> best_score:
            best_score=score
            best_model=tuned_model
            best_model_name=name
            best_model_predict=predictions_after