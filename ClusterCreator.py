{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "86e7cc72-3c1d-40c2-bd1f-e39ced380c6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#TO BE USED AFTER Layer8ResponseCollector\n",
    "#Purpose: cluster the responses of the subnet without any methods of dimensionality reduction. This is solely for the purpose of manually inspecting what pictures are clustering.\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.decomposition import PCA\n",
    "import matplotlib.pyplot as plt\n",
    "from keras.preprocessing.image import load_img \n",
    "\n",
    "#put your subnetresponse csv from the previous code\n",
    "\n",
    "data = pd.read_csv('yoursubnetresponseshere')\n",
    "\n",
    "filenames = data.iloc[1:,0].to_numpy()\n",
    "feat = data.iloc[1:, 1:].to_numpy()\n",
    "kmeans = KMeans(n_clusters=365, random_state=22)\n",
    "kmeans.fit(feat)\n",
    "\n",
    "groups = {}\n",
    "for file, cluster in zip(filenames,kmeans.labels_):\n",
    "    if cluster not in groups.keys():\n",
    "        groups[cluster] = []\n",
    "        groups[cluster].append(file)\n",
    "    else:\n",
    "        groups[cluster].append(file)\n",
    "        \n",
    "def view_cluster(cluster):\n",
    "    \n",
    "    files = groups[cluster]\n",
    "    plt.figure(figsize = (5,5));\n",
    "    num_plots = len(files)\n",
    "    num_rows = (num_plots + 2) // 3 \n",
    "    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))\n",
    "    if num_rows > 1:\n",
    "        axs = axs.flatten()\n",
    "    \n",
    "    if len(files) > 30:\n",
    "        print(f\"Clipping cluster size from {len(files)} to 30\")\n",
    "        files = files[:29]\n",
    "\n",
    "    for index, file in enumerate(files):\n",
    "        img = load_img(f'C:\\\\Users\\\\gkrul\\\\OneDrive\\\\Desktop\\\\NN\\\\P365\\\\val2017\\\\{file}')\n",
    "        img = np.array(img)\n",
    "        axs[index].imshow(img)\n",
    "        \n",
    "    for i in range(num_plots, len(axs)):\n",
    "        axs[i].axis('off')\n",
    "        \n",
    "    plt.tight_layout()\n",
    "        \n",
    "view_cluster(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0c35190-6e9c-4149-ae36-51dfb37aefce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
