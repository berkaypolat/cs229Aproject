#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:04:29 2019

@author: berkaypolat
"""

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('newsData.csv')
  
articleLengths = [len(row.split()) for row in df['Content no HTML']]
plt.hist(articleLengths, bins = 20)
plt.title("NUmber of Words in Articles in Dataset")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.savefig('histogram.png')