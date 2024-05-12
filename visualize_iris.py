import streamlit as st
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Read the dataset into a pandas DataFrame
iris_df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
# iris_df  # columns => sepal_length, sepal_width, petal_length, petal_width, species

# # จะ plot ได้ต้องมี 2 มิติ โดย Iris มี 4 มิติ จะให้ user เลือก มา 2 column แรก สีเปลี่ยนตาม class เพื่อให้เห็น อิทธิพลของ class ก็ได้
axisX = st.selectbox("Select X-axis",
                  iris_df.columns[::-1])# ให้ user เลือก dataset ซึ่งจะต้องมี widget ie. st.selectbox etc ซึ่งไม่เอา column ที่เป็ฯ class อาจจะให้ error เพราะเป็น string

axisY = st.selectbox("Select Y-axis",
                  iris_df.columns[:-1]) # ให้ user เลือก dataset ซึ่งจะต้องมี widget ie. st.selectbox etc

st.write('You selected:', axisX, axisY)
st.scatter_chart(iris_df, x=axisX, y=axisY, color=iris_df.columns[-1]) # discriminant คือกระจายค่อนข้างเดียว project (ฉายไฟเข้า)เข้าแกน ทั้ง 2 แกน ถ้าไม่ทับก็ดี เช่น แกน กับ แกน


