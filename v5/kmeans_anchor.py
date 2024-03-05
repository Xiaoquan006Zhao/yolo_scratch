import os
import numpy as np
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
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(data)
    return kmeans.cluster_centers_

def auto_anchor(num_anchors, label_dir, scales):
    num_clusters = num_anchors
    label_data = read_labels(label_dir)
    anchor_boxes = np.sort(perform_kmeans(label_data, num_clusters), axis=0)

    print("Resulting Anchor Box Sizes:")
    print(anchor_boxes)

    ANCHORS = [[] for _ in range(len(scales))]
    ANCHORS[0] = anchor_boxes.astype(float)
    
    for i in range(1, len(scales)):
        scaled_anchor_boxes = [anchor_box / float(2*i) for anchor_box in anchor_boxes]
        ANCHORS[i].extend(scaled_anchor_boxes)
    
    ANCHORS = [[tuple(subarray) for subarray in outer] for outer in ANCHORS]
    
    return ANCHORS