import seaborn as sns
import streamlit as st
import numpy as np
st.title('Iris Dataset')
df = sns.load_dataset('iris')
st.sidebar.title('Input Parameters')
x = np.array([0.] * 4) # ทำให้เป็นค่า float และมี shape = 4
x.shape

for i, col in enumerate(df.columns):
    if col != 'species':
        x[i] = st.sidebar.slider(col, .9*df[col].min(), 1.1*df[col].max(), df[col].mean()) # .9 กับ 1.1 ที่มันคุณมาเพิ่มตรงนี้คือ แค่ขยายช่วงให้ลากเฉยๆ
st.write(x)
Xtrain = df.iloc[:, :-1].values
Ytrain = df.iloc[:, -1].values
d = np.sum((Xtrain - x)**2, axis=1) # ทำ ยูคลิเดียน distance
st.write('Predicted class:', Ytrain[d.argmin()])

# e พิมพ์ลงใน input ได้