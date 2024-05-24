import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.inspection import DecisionBoundaryDisplay #object ของ matplotlib
from matplotlib import pyplot as plt

# จะ classify ต้อง แบ่ง data
## โดยวิธีนี้จะแบ่งจาก การใช้ np ในการกำหนด index แบบ fix เลย

st.title('Iris Data Classification!')
df = px.data.iris()
itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]
dr = st.selectbox('Select DR method', ['PCA', 't-SNE', 'MDS', 'Isomap'])
DR = {'PCA': PCA, 't-SNE': TSNE, 'MDS': MDS, 'Isomap': Isomap}
X = df.iloc[:, :4]
X = DR[dr](n_components=2).fit_transform(X)
xtrain = X[itrain]
ytrain = df.iloc[itrain, 5]
xtest = X[itest]
ytest = df.iloc[itest, 5]

# interface ของแต่ละตัว เหมือนกัน เลยใช้แบบนี้ได้
K = list(range(1, len(xtrain) + 1))
k = st.select_slider('Select k',
                     options=K)
cls = KNeighborsClassifier(n_neighbors=k)
cls.fit(xtrain, ytrain)
ztest = cls.predict(xtest)
# zproba = cls.predict_proba(xtest)
# zproba
acc = np.sum(ytest == ztest) / len(ytest)
st.write(f'Accuracy = {acc * 100:.2f}%')
col1, col2 = st.columns(2)
disp = DecisionBoundaryDisplay.from_estimator( # ขอบ
    cls,
    xtrain,
    alpha=0.5,
)
disp.ax_.scatter(X[:, 0], # จุด
                 X[:, 1],
                 c=df.iloc[:, 5],
                 edgecolor='k')
col1.pyplot(disp.figure_)

# for label in np.unique(df.iloc[:, 5]):
#     disp = DecisionBoundaryDisplay.from_estimator(
#         cls,
#         xtrain,
#         response_method='predict_proba',
#         class_of_interest=label,
#         alpha=0.5,
#     )
#     disp.ax_.scatter(X[:, 0],
#                      X[:, 1],
#                      c=df.iloc[:, 5],
#                      edgecolor='k')
#     col2.pyplot(disp.figure_)