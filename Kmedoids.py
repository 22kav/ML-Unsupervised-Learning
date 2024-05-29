import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
data=pd.read_csv('D:\Ai&Ml\HeartDisease.csv')
x=data[['ejection_fraction','platelets']].values
def calculate_cost(x,medoids,distance_metric):
    distance_matrix=pairwise_distances(x,medoids,metric=distance_metric)
    return np.sum(np.min(distance_matrix,axis=1))
def pam(x,k,max_iterations=100,distance_metric='euclidean'):
    num_samples,num_features=x.shape
    medoids=x[np.random.choice(num_samples,k,replace=False)]
    for _ in range(max_iterations):
        labels=np.argmin(pairwise_distances(x,medoids,metric=distance_metric),axis=1)
        for i in range(k):
                         current_cost=calculate_cost(x,medoids,distance_metric)
                         non_medoids_idx=np.where(labels!=i)[0]
                         for candidate_medoid in non_medoids_idx:
                             new_medoids=np.copy(medoids)
                             new_medoids[i]=x[candidate_medoid]
                             candidate_cost=calculate_cost(x,new_medoids,distance_metric)
                             if candidate_cost<current_cost:
                                 medoids=np.copy(new_medoids)
    return labels,medoids
k=3
cluster_labels,medoids=pam(x,k)
plt.figure(figsize=(8,6))
for label in set(cluster_labels):
    cluster_data=x[cluster_labels==label]
    plt.scatter(cluster_data[:,0],cluster_data[:,1],label=f"Cluster {label}")
plt.scatter(medoids[:,0],medoids[:,1],marker='x',s=200,c='black',label="Medoids")
plt.xlabel('ejection_fraction')
plt.ylabel('platelets')
plt.title('PAM clustering')
plt.legend()
plt.grid(True)
plt.show()
