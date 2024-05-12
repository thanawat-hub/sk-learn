import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.manifold import MDS, Isomap, TSNE

from sklearn.datasets import load_wine

st.title("Data Visualization")

tmp = load_wine(as_frame=True)
df = tmp['data']
df['target'] = tmp['target']
# df

#PCA
d = st.select_slider("Select PCA dimension",
                     options=[1, 2, 3])
if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :4]) # ouput เป็น numpy array
    # X = PCA(n_components=3).fit_transform(df.iloc[['ชื่อcolumn'],['ชื่อcolumn']])?
    X = pd.DataFrame(X, columns=['x', 'y', 'z']) # ทำกลับมาให้เป็น df
    X['target'] = df['target'] # เอามิติที่ 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
    fig = px.scatter_3d(X,
                        x='x',
                        y='y',
                        z='z',
                        color='target')
    st.plotly_chart(fig)
if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['target'] = df['target'] # เอามิติที่ 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
    fig = px.scatter(X,    #
                    x='x',
                    y='y',
                    color='target')
    st.plotly_chart(fig)
if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target'] # เอามิติที่ 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
    X['y'] = 0
    fig = px.scatter(X, # แต่ engine นี้ รองรับ 2 มิติ เลย ลบ ไม่ได้ ก็กำหนด y ให้ = 0
                     x='x', # ตรงนี้คือการเรียกชื่อ columns
                     y='y', # ตรงนี้คือการเรียกชื่อ columns
                     color='target')
    st.plotly_chart(fig)


# MDS
st.subheader('MDS')
d = st.select_slider("Select MDS dimension",
                     options=[1, 2, 3])
if d == 3:
    X = MDS(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='target')
    st.plotly_chart(fig)
if d == 2:
    X = MDS(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
if d == 1:
    X = MDS(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
# Isomap
st.subheader('Isomap')
d = st.select_slider("Select Isomap dimension",
                     options=[1, 2, 3])
k = st.select_slider("Select Isomap neighbors",
                     options=list(range(3, 20)))
if d == 3:
    X = Isomap(n_components=3, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='target')
    st.plotly_chart(fig)
if d == 2:
    X = Isomap(n_components=2, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
if d == 1:
    X = Isomap(n_components=1, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
# t-SNE
st.subheader('t-SNE')
d = st.select_slider("Select t-SNE dimension",
                     options=[1, 2, 3])
if d == 3:
    X = TSNE(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='target')
    st.plotly_chart(fig)
if d == 2:
    X = TSNE(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
if d == 1:
    X = TSNE(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)