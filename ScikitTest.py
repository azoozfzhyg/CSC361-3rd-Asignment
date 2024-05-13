from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pylab as plt
X, y = load_breast_cancer(return_X_y=True)

mod = KNeighborsRegressor()

mod.fit(X, y)

prediction = mod.predict(X)

plt.scatter(prediction, y)
