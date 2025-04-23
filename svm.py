import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics  import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

iris =datasets.load_iris()
X=iris.data  #Featuur (petal length, petle width etc)
y=iris.target  #Target Species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train) 

y_pred = svm.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n Classification Report: ",classification_report(y_test, y_pred,target_names=iris.target_names))

pca= PCA(n_components=2)
X_pca = pca.fit_transform(X)

svm_2d=SVC(kernel="linear")
svm_2d.fit(X_pca, y)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
h = .02  # Step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', marker='o', s=100)
plt.title("SVM Decision Boundary on Iris Dataset (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()