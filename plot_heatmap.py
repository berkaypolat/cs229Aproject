#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:19:04 2019

@author: berkaypolat
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


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


data = read_similarity_matrix('CSM3.txt').reshape(1224,1224,12)
data1 = data[150:170,150:170,-1]
ax = sns.heatmap(data1, cmap='RdYlGn', linewidth=0.3)
plt.title("Article Correlation With Affinity Propagation")
plt.show()
