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

# Evaluation Metrics
y_prob=model.predict_proba(X_test)[:1]
print(classification_report(y_test,y_pred))

# Confusion Matrix
cm= confusion_matrix(y_test,y_pred)
disp= ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC Score=
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# ROC Curve & AUC
# ROC Curve plots TPR (Recall) vs FPR (False Positive Rate)
# AUC (Area Under Curve): Ranges from 0 to 1 — the higher the better.
# AUC = 0.5 → Random model
# AUC ≈ 1.0 → Excellent model