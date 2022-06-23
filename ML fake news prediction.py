# -*- coding: utf-8 -*-
"""
Created on Wed Jun 8 04:32:37 2022

@author: Ramagiri Nihal
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
news_dataset = pd.read_csv('D:/ML fake news prediction/train.csv')
 
news_dataset.shape
 
news_dataset.head()
 
news_dataset.isnull().sum()
 
 #replacing null values with empty string
news_dataset =news_dataset.fillna('')
 
news_dataset['content'] = news_dataset['author'] +' '+news_dataset['title']
 
print(news_dataset['content'])
 
 #seperating the data and label
X =news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
 
print(X)
print(Y)
 
""" Stemming"""
port_stem = PorterStemmer()
 
def stemming(content):
    stemmed_content = re.sub('[˄a-zA-z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ''.join(stemmed_content)
    return stemmed_content
 
news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])
 
#separating the data and label

X= news_dataset['content'].values
Y = news_dataset['label'].values
print(X)
print(Y)
Y.shape
X.shape

#converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)

"Splitting our dataset to training and test dsta"

X_train,X_test ,Y_train, Y_test = train_test_split(X,Y, test_size =0.2,stratify= Y, random_state=2)

"Training model using Logestic regression"
model = LogisticRegression()

model.fit(X_train, Y_train)

#accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of training data: ', training_data_accuracy)
#accuracy score for training data is 0.5475360576923077

#accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of test data: ', test_data_accuracy)
#accuracy score for test data is 0.5221153846153846

"Making a Predictive System"
X_news = X_test[1]

prediction = model.predict(X_news)
print(prediction)
if (prediction == 1):
    print("The news is Real")
else:
    print("The news is Fake")
    
print(Y_test[0])