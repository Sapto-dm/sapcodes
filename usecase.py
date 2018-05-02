# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:58:47 2018

@author: acer
"""
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer

#code for stemming
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#code for Lemmatizaion
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]



df = pd.read_csv('D:/Research/Genpact/mccusecase1.csv',encoding='iso-8859-1')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tfidf = TfidfVectorizer(min_df=2, norm='l2', ngram_range=(1, 2), stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=False)
count_train = tfidf.fit(df.Desc)
bag_of_words = tfidf.transform(df.Desc)
features =tfidf.get_feature_names()
indices = np.argsort(tfidf.idf_)[::-1]
top_n = 25
top_features = [features[i] for i in indices[:top_n]]
print(top_features)

