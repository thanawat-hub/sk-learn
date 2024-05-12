from sklearn.datasets import load_iris
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score
import numpy as np
tmp = load_iris(as_frame=True)
X = tmp['data']
Y = tmp['target']
clustering = st.selectbox('Select clustering algorithm',
                          ['KMeans', 'DBSCAN', 'Spectral', 'Agglomerative'])
if clustering == 'KMeans':
    k = st.slider('Select k', min_value=2, max_value=10)
    clu = KMeans(n_clusters=k)
if clustering == 'Spectral':
    k = st.slider('Select k', min_value=2, max_value=10)
    clu = SpectralClustering(n_clusters=k)
if clustering == 'Agglomerative':
    k = st.slider('Select k', min_value=2, max_value=10)
    clu = AgglomerativeClustering(n_clusters=k)
if clustering == 'DBSCAN':
    eps = st.slider('Select eps', min_value=0.1, max_value=2.0)
    min_samples = st.slider('Select min_samples', min_value=1, max_value=10)
    clu = DBSCAN(eps=eps, min_samples=min_samples)
clu.fit(X)
if clustering in ['DBSCAN', 'Spectral', 'Agglomerative']:
    Z = clu.labels_
    k = np.max(clu.labels_) + 1
    st.write(f'Number of clusters = {k}')
else:
    Z = clu.predict(X)
randind = rand_score(Y, Z)
st.write(f'Rand index = {randind:.2f}')
plt.figure()
for i in range(k):
    plt.plot(X.iloc[Z == i, 0], X.iloc[Z == i, 1], '.')
st.pyplot(plt)