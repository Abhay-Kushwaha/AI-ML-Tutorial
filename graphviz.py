from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
# from sklearn.metrics import (
#     confusion_matrix, classification_report, ConfusionMatrixDisplay,
#     roc_curve, roc_auc_score, accuracy_score, precision_score,
#     recall_score, f1_score
# )

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# Data
X = [
    [1, 1, 0],  # Cloudy, Rain forecast
    [1, 0, 1],  # Cloudy, No rain, Long walk
    [0, 1, 1],  # Not cloudy, Rain forecast
    [0, 0, 0],  # Clear and dry
]
y = [1, 1, 1, 0]  # 1 = Bring umbrella, 0 = Don't bring

# Feature and target names
feature_names = ['Cloudy', 'RainForecast', 'LongWalk']
target_names = ['No', 'Yes']

# Train model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Export as DOT file and render
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("umbrella_decision_tree", format="png", cleanup=True)  # Saves as umbrella_decision_tree.png
graph.view()  # Opens the image file