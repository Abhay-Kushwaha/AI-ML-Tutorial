import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

X, Y= make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)

#Apply Ridge Regression
ridge= Ridge(alpha=1.0) 
ridge.fit(X_train,Y_train)
y_pred_ridge= ridge.predict(X_test)

#Apply Lasso Regression
lasso= Lasso(alpha=1.0) 
lasso.fit(X_train,Y_train)
y_pred_lasso= lasso.predict(X_test)

print("Ridge MSE: ", mean_squared_error(Y_test, y_pred_ridge))
print("Lasso MSE: ", mean_squared_error(Y_test, y_pred_lasso))

plt.scatter(X_test, Y_test, color="blue", label="Actual data")
plt.plot(X_test, y_pred_ridge, color="red", label="Ridge Regression")
plt.plot(X_test, y_pred_lasso, color="green", label="Lasso Regression")
plt. legend()
plt.title("Ridge VS Lasso")
plt.grid()
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.show()