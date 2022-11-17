import csv
import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


y = []

with open("data/iris.data") as f:
    for line in f:
        line = line.strip().split(',')
        if line == ['']:
            continue
        y.append(line[-1])

y = LabelEncoder().fit_transform(y)

with open("data/predict.txt") as f:
    y_pred = [int(x) for x in f.read().split('\n')]


precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')

with open("data/metrics.json", "w") as fd:
    json.dump(
        {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        },
        fd
    )

with open("data/classes.csv", "w", newline='') as result:
    writer = csv.writer(result)
    writer.writerow(["actual", "predicted"])
    writer.writerows(zip(y, y_pred))