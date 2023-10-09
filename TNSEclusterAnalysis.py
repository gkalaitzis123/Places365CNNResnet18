#TO BE USED AFTER Layer8ResponseCollector
#Purpose: cluster the responses of the subnet with t-SNE. Then visualize using scatterplot.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img 
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

#put your subnetresponse csv from the previous code

data = pd.read_csv(yourpath)

#separates the file names (first column) from the rest of the data

filenames = data.iloc[1:,0].to_numpy()
feat = data.iloc[1:, 1:].to_numpy()
 
#tsne on the dataset, need 2 dimensions for plot

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
x = tsne.fit_transform(feat)

#performs kmeans clustering on the dataset, change the amount of clusters to whatever you want

kmeans = KMeans(n_clusters=50, random_state=22)
label = kmeans.fit_predict(x)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(x[label == i , 0] , x[label == i , 1] , label = i)
#plt.savefig(f'TSNEscatter.png')
plt.show()

#Statistical analysis of clustering performance

db = davies_bouldin_score(x, label)
print(db)
sc = silhouette_score(x, label)
print(sc)
ch = calinski_harabasz_score(x, label)
print(ch)

#CODE BELOW IS JUST FOR VIEWING CREATED CLUSTERS

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

def view_cluster(cluster):
    
    files = groups[cluster]
    plt.figure(figsize = (5,5));
    num_plots = len(files)
    num_rows = ((num_plots + 2) // 5) + 1
    fig, axs = plt.subplots(num_rows, 5, figsize=(12, 4 * num_rows))
    if num_rows > 1:
        axs = axs.flatten()
    
    for index, file in enumerate(files):
        img = load_img(yourpath)
        img = np.array(img)
        axs[index].imshow(img)
        
    for i in range(num_plots, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    
    #plt.savefig(yourplot)
    
view_cluster(2)
