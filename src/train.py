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

clf = DecisionTreeClassifier().fit(X, y)
y_pred = clf.predict(X)
with open("data/predict.txt", 'w') as f:
    print(*y_pred, sep='\n', end='', file=f)
