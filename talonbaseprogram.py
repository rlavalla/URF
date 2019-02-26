import time
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer   # “Term Frequency times Inverse Document Frequency”
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB       # Multinomial Naive Bayes, supposedly goes well with the data from the transformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sklearn.neighbors
import sklearn.neural_network


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
LogisticRegression(solver='lbfgs')


t0 = time.time()


comments = []


#figure out average length of user comment

#set min and max length, min start with 5 words,



counter=1
with open("DATA/Top10IndividualUsers/threeusers.json") as f:
    for line in f:
        print(counter)
        comments.append(json.loads(line))
        counter += 1


texts = [comment['body'] for comment in comments]
authors = [comment['author'] for comment in comments]



vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)
print("Printing Vector Shape...")
print(vectors.shape)

X_train1, X_test1, y_train1, y_test1 = train_test_split(vectors, authors, test_size=0.2, train_size=0.8, random_state=123, shuffle=True)

print("Printing X_train1 and X_test1 shape...")
print(X_train1.shape, X_test1.shape)

svm = LinearSVC()
svm.fit(X_train1, y_train1)

svm = LinearSVC()
svm.fit(X_train1, y_train1)
predictions = svm.predict(X_test1)

print("Training/Evaluating Bernoulli Naive Bayes model...")
bnb_model = BernoulliNB().fit(X_train1, y_train1)
bnbmodelpredict = bnb_model.predict(X_test1)
print("BernoulliNB accuracy, on validation data: ", classification_report(y_test1, bnbmodelpredict), bnb_model.score(X_test1, y_test1))


# Mulitnomial Naive Bayes model
print("Training/Evaluating Multinomial Naive Bayes model...")
mnb_model = MultinomialNB().fit(X_train1, y_train1)
mnbmodelpredict = mnb_model.predict(X_test1)
print("MultinomialNB accuracy, on validation data: ", classification_report(y_test1, mnbmodelpredict), mnb_model.score(X_test1, y_test1))

# Logistic Regression Model
print("Training/Evaluating Logistic Regression model...")
lr_model = LogisticRegression().fit(X_train1, y_train1)
lrmodelpredict = lr_model.predict(X_test1)
print("Logistic Regression accuracy, on validation data: ", classification_report(y_test1, lrmodelpredict), lr_model.score(X_test1, y_test1))

print("Training/Evaluating Nearest Neighbors...")
nearestneighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=40)
nearestneighbors.fit(X_train1, y_train1)
nnpredict = nearestneighbors.predict(X_test1)
print("Nearest Neighbors accuracy, on validation data:", classification_report(y_test1, nnpredict), nearestneighbors.score(X_test1, y_test1))

print("Training/Evaluating MLP Neural Net...")
neuralnetwork = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=200, random_state=123)
neuralnetwork.fit(X_train1, y_train1)
neuralnpredict = neuralnetwork.predict(X_test1)
print("Neural Network accuracy, on validation data: ", classification_report(y_test1, neuralnpredict), neuralnetwork.score(X_test1, y_test1))

t1 = time.time()

print("time to run: ", t1-t0)
