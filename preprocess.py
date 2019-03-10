#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:47:43 2019

@author: berkaypolat
"""

import pandas as pd
import string

df = pd.read_csv('newstories.csv')
df = df.drop(['Date','Status','Post Type'], axis = 1)

table = str.maketrans(dict.fromkeys(string.punctuation + '\r\n“”’', " "))

df['Content no HTML'] = df['Content no HTML'].str.translate(table)

personGroupList = df['Persons / Groups'].str.split('|')
topicList = df['Topics'].str.split('|')
regionList = df['Places'].str.split('|')
themeList = df['Themes'].str.split('|')

#drop Nan values
personGroupList = personGroupList.dropna()
topicList = topicList.dropna()
regionList = regionList.dropna()
themeList = themeList.dropna()

#turn pandas Series to python list
entityList = personGroupList.tolist()
topicList = topicList.tolist()
regionList = regionList.tolist()
themeList = themeList.tolist()

#flatten out the list of lists
entityList = [item for sublist in entityList for item in sublist]
topicList = [item for sublist in topicList for item in sublist]
regionList = [item for sublist in regionList for item in sublist]
themeList = [item for sublist in themeList for item in sublist]


entityListSet = set(entityList)
topicList = list(set(topicList))
regionList = list(set(regionList))
themeList = list(set(themeList))

entityTags = list(entityListSet)

#write all tags to a txt file
with open('labels/entityTags.txt', 'w') as f:
    for item in entityTags:
        f.write("%s\n" % item)

with open('labels/topicTags.txt', 'w') as f:
    for item in topicList:
        f.write("%s\n" % item)

with open('labels/regionTags.txt', 'w') as f:
    for item in regionList:
        f.write("%s\n" % item)

with open('labels/themeTags.txt', 'w') as f:
    for item in themeList:
        f.write("%s\n" % item)

encodedList = []
for row in df['Content no HTML']:
    rowList = row.split()
    encodedRow = []
    for word in rowList:
        if (word in entityListSet):
            encodedRow.append('PER')
        else:
            encodedRow.append('0')
    outputStr = ' '.join(encodedRow)
    encodedList.append(outputStr)

df['EncodedTags'] = encodedList
   

df.to_csv("newsData.csv")
