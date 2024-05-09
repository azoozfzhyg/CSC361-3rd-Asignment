from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
import matplotlib as plt
X, y = load_breast_cancer(return_X_y=True)

mod = KNeighborsRegressor()

mod.fit(X, y)

pred = mod.predict(X)

plt.scatter(pred, y)
