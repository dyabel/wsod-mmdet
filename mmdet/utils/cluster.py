from mmdet.utils import iou
from sklearn.cluster import KMeans
import numpy as np
def calculate_distance(centroid,boxes):
    box_centroid_distances = []
    for box in boxes:
        distances = []
        for center in centroid:
            distance = iou(box.unsqueeze(0),center.unsqueeze(0))[0][0]
            distances.append(distance)
        box_centroid_distances.append(distances)
    return box_centroid_distances
kmeans = KMeans(n_clusters=10,random_state=0)
def average(cluster):
    return np.mean(cluster,0).tolist()

def k_means(boxes,n_clusters=5):
    centroid = boxes[:n_clusters,:]
    flag = True
    labels = list(range(len(boxes)))
    while flag:
        flag = False
        box_centroid_distances = calculate_distance(centroid,boxes)
        clusters = {i:[] for i in range(n_clusters)}
        for i,distances in enumerate(box_centroid_distances):
            assigned_center = np.argmin(distances)
            if labels[i] != assigned_center:
                labels[i] = assigned_center
                flag = True
            clusters[np.argmin(distances)] = i
        for i,cluster in clusters.items():
            centroid[i] = average(cluster)
    return centroid



