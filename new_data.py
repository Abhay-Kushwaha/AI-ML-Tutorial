import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from graphviz import Source

# Step 1: Create dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast',
                'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
                    'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild',
                    'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'Played': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
               'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
               'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Encode categorical features
le = LabelEncoder()
encoded_df = df.apply(le.fit_transform)

X = encoded_df.drop('Played', axis=1)
y = encoded_df['Played']

# Step 3: Train Decision Tree with Gini Index
clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

# Step 4: Plot the tree
feature_names = X.columns
class_names = le.fit(df['Played']).classes_
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = Source(dot_data)
graph.render("football_decision_tree", view=True, format='png')  # Saves and opens PNG

# Step 5: Accept user input
print("\n--- Test a New Sample ---")
user_input = {}
for col in feature_names:
    val = input(f"Enter {col} (options: {df[col].unique().tolist()}): ")
    user_input[col] = val

# Step 6: Encode user input
test_input = pd.DataFrame([user_input])
for col in test_input.columns:
    le.fit(df[col])
    test_input[col] = le.transform(test_input[col])

# Step 7: Predict
prediction = clf.predict(test_input)
le.fit(df['Played'])
result = le.inverse_transform(prediction)
print("\nPrediction for the given input:", result[0])