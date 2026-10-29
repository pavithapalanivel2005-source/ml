import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(r"C:\ml\Iris.csv")

x = data[["SepalLengthCm", "SepalWidthCm"]].values

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
c = kmeans.fit_predict(x)
ce = kmeans.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=c, cmap="viridis")
plt.scatter(ce[:, 0], ce[:, 1], c="red", marker="X", s=200)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
