# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 04:32:37 2022

@author: Ramagiri Nihal

resource from www.kaggle.com/competitions/fake-news/data
1: Fake news
0: real news
"""

"""Importing dependencies."""
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

import nltk
nltk.download('stopwords')
#prnting the stopwords in English
print(stopwords.words('english'))

""" Data preprocessing"""
#loading the dataset to a pandas DtataFrame
news_dataset = pd.read_csv('D:/ML fake news prediction/train.csv')
#to get number of rows and columns
news_dataset.shape
#print first 5 rows of dataset
news_dataset.head()

#counting the number of missing values in the dataset
news_dataset.isnull().sum()
#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

