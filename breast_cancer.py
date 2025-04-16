from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

data = load_breast_cancer()
X = data.data 
y = data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model=LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print(y_pred)

# Evaluation Metrics
y_prob=model.predict_proba(X_test)[:, 1]
print(classification_report(y_test,y_pred))

# Confusion Matrix
cm= confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
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
