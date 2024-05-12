import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import numpy as np
import pandas as pd

df = px.data.iris()

# จะ classify ต้อง แบ่ง data
## โดยวิธีนี้จะแบ่งจาก การใช้ np ในการกำหนด index แบบ fix เลย

st.title('Iris Data Classification!')
df = px.data.iris()
itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]

xtrain = df.iloc[itrain, :4]
ytrain = df.iloc[itrain, 4]
xtest = df.iloc[itest, :4]
ytest = df.iloc[itest, 4]


K = list(range(1, len(xtrain) + 1))
k = st.select_slider('Select k',
                     options=K)
cls = KNeighborsClassifier(n_neighbors=k)
cls.fit(xtrain, ytrain)
ztest = cls.predict(xtest)
acc = np.sum(ytest == ztest) / len(ytest)
st.write(f'Accuracy = {acc * 100:.2f}%')

# df = pd.DataFrame(columns=['ytest', 'ztest'])
# df['ytest'] = ytest
# df['ztest'] = ztest

# # ----
# # โจทย์คือ ไม่ต้อง เลื่อนค่า k  เนื่องจาก data น้อย เลยจะดูทุกค่า k เป็นรูปมาเลย
# # axis x = k , y = acc
# # ต้องเทรนทุกครั้ง แล้วหาทุกค่า ดังนั้นจึง for ด้วย k
# # แล้ว plot graph

ACC = []
for k in K:
    cls = KNeighborsClassifier(n_neighbors=k)
    cls.fit(xtrain, ytrain)
    ztest = cls.predict(xtest)
    ACC.append(np.sum(ytest == ztest) / len(ytest))
df = pd.DataFrame(columns=['k', 'acc'])
df['k'] = K
df['acc'] = ACC
fig = px.line(df, x='k', y='acc')
st.plotly_chart(fig)
