#TO BE USED AFTER Layer8ResponseCollector
#Purpose: cluster the responses of the subnet without any methods of dimensionality reduction. This is solely for the purpose of manually inspecting what pictures are clustering.

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from itertools import cycle

#put your subnetresponse csv from the previous code

data = pd.read_csv(yourpath)

# Assuming your features are in columns 1 to N (adjust the column indices accordingly)
filenames = data.iloc[1:,0].to_numpy()
feat = data.iloc[:, 1:]

#Choose PCA or t-SNE with the data

#pca = PCA(n_components=2, random_state=22)
#pca.fit(feat)
#x = pca.transform(feat)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
x = tsne.fit_transform(feat)

# Create the affinity matrix (using Gaussian RBF kernel)
affinity_matrix = np.exp(-0.5 * np.square(np.linalg.norm(x[:, None] - x, axis=2)))

# Perform spectral clustering
n_clusters = 50  # Set the number of clusters
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
cluster_labels = spectral_clustering.fit_predict(affinity_matrix)

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
#plt.savefig(f'specPCA/TSNEscatter.png')
plt.show()
