#TO BE USED AFTER Layer8ResponseCollector
#Purpose: cluster the responses of the subnet without any methods of dimensionality reduction. The second part (after the grouping) is solely for the purpose of manually inspecting what pictures are clustering.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img 

#put your subnetresponse csv from the previous code

data = pd.read_csv('yoursubnetresponseshere')

#separates the file names (first column) from the rest of the data

filenames = data.iloc[1:,0].to_numpy()
feat = data.iloc[1:, 1:].to_numpy()

#performs kmeans clustering on the dataset, change the amount of clusters to whatever you want

kmeans = KMeans(n_clusters=yourclusteramthere, random_state=22)
kmeans.fit(feat)

#creates a dictionary that groups together each file in a cluster into a list with their cluster number as the key

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

#function for viewing the clusters using matplotlib, you can change all of the specifications, current ones lead to rows of 3 pictures being displayed.

def view_cluster(cluster):
    
    files = groups[cluster]
    plt.figure(figsize = (5,5));
    num_plots = len(files)
    num_rows = (num_plots + 2) // 3 #change the 3 to what you want for rows, also change the 3 in the row below
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
  
    if num_rows > 1:
        axs = axs.flatten()

    for index, file in enumerate(files):
        img = load_img(f'pathtofileshere{file}')
        img = np.array(img)
        axs[index].imshow(img)
        
    for i in range(num_plots, len(axs)):
        axs[i].axis('off')
        
    plt.tight_layout()

#View whatever cluster you want

view_cluster(yourclusterintegerhere)
