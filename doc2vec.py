#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:44:50 2019

@author: berkaypolat
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import pandas as pd

df = pd.read_csv('newsData.csv')

docs = []

#build the list of TaggedDocument objects
for index, row in df.iterrows():
    words = row['Content no HTML'].split()
    if(row['Persons / Groups'] == row['Persons / Groups']):
        entityList = row['Persons / Groups'].split('|')
        tags = entityList
    if(row['Topics'] == row['Topics']):
        topicList = row['Topics'].split('|')
        tags.extend(topicList)
    if(row['Places'] == row['Places']):
        regionList = row['Places'].split('|')
        tags.extend(regionList)
    if(row['Themes'] == row['Themes']):
        themeList = row['Themes'].split('|')
        tags.extend(themeList)
    tags.append('DOC_' + str(index))
    docs.append(TaggedDocument(words,tags))

cores = multiprocessing.cpu_count()

#instantiate models
models = [
        Doc2Vec(dm=0, hs = 0, negative =5, vector_size=100, epochs=40, min_count = 3, dbow_words = 1, workers = cores),
        Doc2Vec(dm=1, dm_mean=1, hs=0, negative=5, vector_size=100, epochs=40, min_count = 3, dbow_words = 1, workers = cores)
        ]

#build vocab for both model
models[0].build_vocab(docs)
models[1].reset_from(models[0])

for model in models:
    model.train(docs,total_examples = model.corpus_count, epochs=model.epochs)    
    
    
    
    
