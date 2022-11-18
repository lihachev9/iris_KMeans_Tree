import csv
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


with open("data/iris.data") as f:
    y = [line.split(',')[-1] for line in f.read().split('\n') if line]

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