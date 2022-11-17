from sklearn.cluster import KMeans


X = []

with open("data/iris.data") as f:
    for line in f:
        line = line.strip().split(',')
        if line == ['']:
            continue
        X.append(list(map(float, line[:-1])))

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_pred = kmeans.predict(X)

with open("data/predict.txt", 'w') as f:
    print(*y_pred, sep='\n', end='', file=f)
