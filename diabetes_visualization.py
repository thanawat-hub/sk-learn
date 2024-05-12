import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.datasets import load_diabetes

# st.title("Data Visualization")
#
# diabetes = load_diabetes() # วิธีที่เขาทำกันคือ ดูค่าจากปุุ่ม debug
# # diabetes.data # raw data
# # diabetes.feature_names # 0:"age", 1:"sex" ... 9:"s6"
#
# # ask ----> chat-gpt
# # # Convert the NumPy array to a DataFrame
# # diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# #
# # # Add the target variable to the DataFrame
# # diabetes_df['target'] = diabetes.target
# # # diabetes_df
# # ------> or read on document
# df_diabetes = load_diabetes(as_frame=True)
#
#
# # PCA
# # d = st.select_slider("Select PCA dimension",
# #                      options=[1, 2, 3])
# # if d == 3:
# #     X = PCA(n_components=3).fit_transform(diabetes_df.iloc[:, :4])  # ouput เป็น numpy array
# #
# #     X = pd.DataFrame(X, columns=['x', 'y', 'z']) # ทำกลับมาให้เป็น df


st.title('Data Visualization!')
tmp = load_diabetes(as_frame=True)
df = tmp['data']
df['target'] = tmp['target']
# PCA
d = st.select_slider("Select PCA dimension",
                     options=[1, 2, 3])
if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['target'] = df['target']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='target')
    st.plotly_chart(fig)
if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['target'] = df['target']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)
if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['target'] = df['target']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='target')
    st.plotly_chart(fig)