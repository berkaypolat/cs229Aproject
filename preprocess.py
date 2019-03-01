#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:47:43 2019

@author: berkaypolat
"""

import pandas as pd
import string

df = pd.read_csv('newstories.csv')
df = df.drop(['Date'], axis = 1)

table = str.maketrans(dict.fromkeys(string.punctuation + '\r\n“”’', " "))

df['Content no HTML'] = df['Content no HTML'].str.translate(table)

personGroupList = df['Persons / Groups'].str.split('|')
personGroupList = personGroupList.dropna()
entityList = personGroupList.tolist()
entityList = [item for sublist in entityList for item in sublist]

entityTags = list(set(entityList))

#write all tags to a txt file
with open('entityTags.txt', 'w') as f:
    for item in entityTags:
        f.write("%s\n" % item)

df.to_csv("newsData.csv")
