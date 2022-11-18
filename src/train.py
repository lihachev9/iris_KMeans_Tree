import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


X = []
y = []

with open("data/iris.data") as f:
    for line in f:
        line = line.strip().split(',')
        if line == ['']:
            continue
        X.append(list(map(float, line[:-1])))
        y.append(line[-1])

y = LabelEncoder().fit_transform(y)

kmeans = DecisionTreeClassifier().fit(X, y)
y_pred = kmeans.predict(X)
with open("data/predict.txt", 'w') as f:
    print(*y_pred, sep='\n', end='', file=f)

feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
feature_importance = kmeans.feature_importances_

with open("data/importances.csv", "w", newline='') as result:
    writer = csv.writer(result)
    writer.writerow(["feature_name", "feature_importance"])
    writer.writerows(zip(feature_name, feature_importance))
