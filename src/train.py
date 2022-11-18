from sklearn.cluster import KMeans


with open("data/iris.data") as f:
    X = [list(map(float, line.split(',')[:-1])) for line in f.read().split('\n') if line]

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_pred = kmeans.predict(X)

with open("data/predict.txt", 'w') as f:
    print(*y_pred, sep='\n', end='', file=f)
