import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.datasets import load_diabetes

st.title('Data Visualization!')
df = pd.read_csv("C:/Users/66825/Desktop/Pim-non/sk-learn/features_raw.csv")
# PCA
df #
df.columns  # 10
df.iloc[:, :-1]

d = st.select_slider("Select PCA dimension",
                     options=[1, 2, 3])
if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :-1])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    # X['target'] = df['target']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z')
    st.plotly_chart(fig)

if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :-1])
    X = pd.DataFrame(X, columns=['x', 'y'])
    # X['target'] = df['target']
    fig = px.scatter(X,
                  x='x',
                  y='y',)
    st.plotly_chart(fig)
if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :-1])
    X = pd.DataFrame(X, columns=['x'])
    # X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',)
    st.plotly_chart(fig)