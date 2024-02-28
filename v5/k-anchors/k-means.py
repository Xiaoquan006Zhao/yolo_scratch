import os
import numpy as np
import config
from sklearn.cluster import KMeans

def read_labels(folder_path):
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    label_data = line.strip().split(' ')
                    width = float(label_data[-2])
                    height = float(label_data[-1])
                    labels.append([width, height])
    return np.array(labels)

def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.cluster_centers_

num_clusters = config.num_anchors
label_data = read_labels(config.label_dir)
anchor_boxes = perform_kmeans(label_data, num_clusters)

print("Resulting Anchor Box Sizes:")
print(anchor_boxes)
