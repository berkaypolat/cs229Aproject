#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:50:06 2019

@author: berkaypolat
"""
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN, AffinityPropagation

def read_similarity_matrix(filename):
    with open(filename, "r") as f:
        if f.mode != "r":
            raise FileNotFoundError("File could not be read.")
        
        sim_matrix = []
        lines = f.readlines()
        for line in lines:
            sim_row = []
            for num in line.split(","):
                sim_row.append(float(num))
            sim_matrix.append(sim_row)
        return np.array(sim_matrix)
    
sim_array = read_similarity_matrix('copy_similarityMatrix.txt')


"""
Performs Spectral Clustering
Input arguments:
    - matrix: precomputed similarity(affinity) matrix
    - num_clusters_list: a list of clusters numbers chosen for the dimension
      of the projection space
The returned clustering attributes are:
    - labels_ : shape of  [n_samples]
                Labels of each point
"""
def applySpectralClustering(matrix, num_clusters_list):
    clustering_labels = []
    for num in num_clusters_list:
        clustering = SpectralClustering(n_clusters = num, affinity = 'precomputed', assign_labels = 'discretize',
                                        n_init=10).fit(matrix)
        clustering_labels.append(clustering)
    return clustering_labels



"""
Performs DBSCAN Clustering from distance matrix
Input arguments:
    - matrix: precomputed similarity(affinity) matrix
    - max_distance_limit_list: a list of prechosen the maximum distances between 
      two samples for them to be considered as in the same neighborhood
The returned clustering attributes are:
    - core_sample_indices_ : shape of [n_core_samples]
                             Indices of core samples
    - components_ : shape of [n_core_samples, n_features]
                    Copy of each core sample found by training
    - labels_ : shape of  [n_samples]
                Noisy samples are given the label -1
"""
def applyDBSCAN(matrix, max_distance_limit_list = [0.5]):
    clustering_labels = []
    for num in max_distance_limit_list:
        clustering = DBSCAN(eps=num, metric= 'precomputed', min_samples=5, algorithm='auto',
                            leaf_size = 30).fit(matrix)
        clustering_labels.append(clustering)
    return clustering_labels


"""
Performs Affinity Propagation Clustering of data.
Input arguments:
    - matrix: precomputed similarity matrix
    - max_iter: max iterations
    - converge_iter: # of iterations with no change that stops the convergence
The returned clustering attributes are:
    - cluster_center_indices_ : shape of (n_clusters,)
    - labels_: shape of (n_samples,)
    - affinity_matrix_
    - n_iter_: number of iterations taken to converge
"""
def applyAffinityPropagatiion(matrix, max_iter=200, converge_iter=15):
    clustering = AffinityPropagation(damping=0.5, max_iter=max_iter,convergence_iter=converge_iter, 
                                     affinity='precomputed', verbose=False ).fit(matrix)
    return clustering
    
    
    
    
    
    