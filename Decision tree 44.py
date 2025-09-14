import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


random.seed(42)
data = pd.DataFrame({
    'Age': [random.randint(20, 60) for _ in range(100)],
    'Job': [random.choice([0,1,2,3,4]) for _ in range(100)],
    'Balance': [random.randint(-1000, 5000) for _ in range(100)],
    'Duration': [random.randint(30, 3000) for _ in range(100)],
    'Campaign': [random.randint(1, 5) for _ in range(100)],
    'Pdays': [random.randint(-1, 10) for _ in range(100)],
    'Poutcome': [random.choice([0,1,2]) for _ in range(100)],
    'Day': [random.randint(1,31) for _ in range(100)],
    'Month': [random.randint(1,12) for _ in range(100)],
    'Contact': [random.choice([0,1]) for _ in range(100)],  # Will be excluded from tree
    'Y': [random.choice([0,1]) for _ in range(100)]
})


X = data.drop(['Y','Contact'], axis=1)  # Exclude 'Contact' column from tree
y = data['Y']


clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)


dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=['No','Yes'],
    filled=True,
    rounded=True,
    label='none',    # Removes samples/value/class
    impurity=False   # Removes entropy/gini
)

graph = graphviz.Source(dot_data)
graph.render("large_clean_decision_tree", format="png", cleanup=True)
graph.view()  # Opens the image
print("Decision tree image saved as 'large_clean_decision_tree.png'")


data['Predicted_Y'] = clf.predict(X)
data.to_csv("processed_large_data_with_predictions.csv", index=False)
print("CSV saved as 'processed_large_data_with_predictions.csv'")