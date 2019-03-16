#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:50:06 2019

@author: berkaypolat
"""
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN, AffinityPropagation
from scipy import spatial

def find_similarity_score(article1, article2):
    return spatial.distance.cosine(article1, article2)

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

"""
Uses the similarity matrix to retrieve clustering results from 3 different
clustering algorithms implemented above.
Returns a list of clustering objects that contains labels (and other properties) for each algorithm
"""
def get_cluster_results(filename):
    sim_array = read_similarity_matrix('copy_similarityMatrix.txt')
    num_clusters_list = [i for i i range(10,20)]
    spectral_clustering = applySpectralClustering(sim_array, num_clusters_list)
    dbscan_clustering = applyDBSCAN(sim_array)
    affprop_clustering = applyAffinityPropagatiion(sim_array)
    clustering_objects = [spectral_clustering, dbscan_clustering, affprop_clustering]
    return clustering_objects

"""
Uses the resulting labels from each clustering algorithm to build a 3-dimensional scores array.
Each article is compared with every other article that is assigned in the same cluster and their
similarity scores are recorded.
The resulting matrix is (n,n,3) matrix where n is the number of total articles
"""
def build_scores_matrix(clustering_objects, embeddings):
    cluster_scores_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0], len(clustering_objects)))

    for i in range(len(clustering_objects)):
        labels = clustering_objects[i].labels_
        scores_matrix = []
        for i in range(len(labels.shape[0])):
            row = []
            cluster = labels[i]
            for j in range(len(labels.shape[0])):
                if(i == j):
                    row.append(1)
                elif (labels[j] == cluster):
                    score = find_similarity_score(embeddings[i], embeddings[j])
                    row.append(score)
                else:
                    row.append(0)
            score_compare_matrix.append(row)
        score_array = np.array(score_compare_matrix)
        cluster_scores_matrix(:,:,i) = score_array

    return cluster_scores_matrix
