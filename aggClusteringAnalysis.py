#TO BE USED AFTER Layer8ResponseCollector
#Purpose: cluster and analyze responses using agglomerative clustering.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from itertools import cycle
from sklearn.cluster import AgglomerativeClustering

#put your subnetresponse csv from the previous code

data = pd.read_csv('C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\\P365\\subnetResponses.csv')

# Assuming your features are in columns 1 to N (adjust the column indices accordingly)
filenames = data.iloc[1:,0].to_numpy()
feat = data.iloc[:, 1:]

#Choose PCA or t-SNE with the data

#pca = PCA(n_components=2, random_state=22)
#pca.fit(feat)
#x = pca.transform(feat)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
x = tsne.fit_transform(feat)

# Perform agglomerative clustering
n_clusters = 50  # Set the number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
cluster_labels = agg_clustering.fit_predict(x)

#Statistical tests to check how good the clustering is

db = davies_bouldin_score(x, cluster_labels)
print(db)
sc = silhouette_score(x, cluster_labels)
print(sc)
ch = calinski_harabasz_score(x, cluster_labels)
print(ch)

# Define a color cycle for cluster colors
colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Create a scatter plot with color-coded clusters and legend
plt.figure(figsize=(8, 6))
for cluster_num in range(n_clusters):
    cluster_points = x[cluster_labels == cluster_num]
    color = next(colors)
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num}', color=color, marker='o', s=50)
plt.savefig(f'aggTSNEscatter.png')
plt.show()
