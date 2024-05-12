import streamlit as st
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

tmp = load_iris(as_frame=True)
X = tmp['data']
Y = tmp['target']
itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]
xtrain = X.iloc[itrain, :]
ytrain = Y.iloc[itrain]
xtest = X.iloc[itest, :]
ytest = Y.iloc[itest]
classifier = st.selectbox('Select classifier',
                              ['SVM', 'Tree', 'Random Forest', 'Naive Bayes'])

CLS = {'SVM': SVC,
       'Tree': DecisionTreeClassifier,
       'Random Forest': RandomForestClassifier,
       'Naive Bayes': GaussianNB}
if classifier == 'SVM':
    kernel = st.selectbox('Select kernel',
                          ['linear', 'poly', 'rbf', 'sigmoid'])
    C = st.select_slider('Select C',
                         options=[0.1, 1, 10, 100, 1000, 10000])
    cls = SVC(kernel=kernel, C=C)
if classifier == 'Tree':
    criterion = st.selectbox('Select criterion',
                             ['gini', 'entropy'])
    max_depth = st.slider('Select max_depth',
                          min_value=1,
                          max_value=10)
    cls = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
if classifier == 'Random Forest':
    n_estimators = st.slider('Select n_estimators',
                             min_value=1,
                             max_value=100)
    cls = RandomForestClassifier(n_estimators=n_estimators)
if classifier == 'Naive Bayes':
    cls = GaussianNB()
cls.fit(xtrain, ytrain)
ztest = cls.predict(xtest)
acc = np.sum(ytest == ztest) / len(ytest)
st.write(f'Accuracy = {acc * 100:.2f}%')

