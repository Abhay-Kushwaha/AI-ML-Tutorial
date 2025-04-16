import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay)

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print(y_pred)
