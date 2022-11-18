import csv
import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def cluster2label(y, y_pred):
    clusters = defaultdict(list)
    num_clusters = len(set(y_pred))
    classes = list(set(y))
    num_classes = len(classes)

    for idx, c in enumerate(y_pred):
        clusters[c].append((idx, y_pred[idx]))

    cluster_label_counts = dict()
    replcae_class = {}

    for c in range(num_clusters):
        cluster_label_counts[c] = [0] * num_classes
        instances = clusters[c]
        for i, _ in instances:
            cluster_label_counts[c][y[i]] += 1

        a = cluster_label_counts[c]
        cluster_label_idx = max(range(len(a)), key = lambda x: a[x])
        cluster_label = classes[cluster_label_idx]
        replcae_class[c] = cluster_label
    for idx, c in enumerate(y_pred):
        y_pred[idx] = replcae_class[c]
    return y_pred


with open("data/iris.data") as f:
    y = [line.split(',')[-1] for line in f.read().split('\n') if line]

y = LabelEncoder().fit_transform(y)

with open("data/predict.txt") as f:
    y_pred = [int(x) for x in f.read().split('\n')]

y_pred = cluster2label(y, y_pred)

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
