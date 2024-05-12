import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Datasets")

ds = st.selectbox("Select Dataset",
                  sns.get_dataset_names())

df = sns.load_dataset(ds)

corr = df.corr(numeric_only=True)# ดึงตัวเลขมาทำ corr โดยใช้ เพียสัน
fig = plt.figure() # numberical atl+f8 # ตั้งชื่อ ว่าจะมีอะไรเข้าไปใส่
sns.heatmap(corr, annot=True) #

# fig # magic เป็น image  ซึ่งจะ interative ไม่ได้

# native streamlit
## ให้ลองใช้ native streamlit -> area chart คล้ายกับ seabron ที่ใช้ df เป็น source input เข้า
## lat long Thai
lat = np.random.randint(130000, 1005000, 100) / 10000
long = np.random.randint(1005000, 1010000, 100) / 10000
df_num = pd.DataFrame(np.vstack([lat, long]).T,
                      columns=['lat', 'lon'])

# st.area_chart(df_num)
# st.map(df_num) # ไปดึงมาจาก open stress map
# st.bar_chart(df_num)
# st.line_chart(df_num)
# st.scatter_chart(df_num)

# thrid party
# เอาแตร โบเก pyplot plotly

# โจทย์
# จะ plot ได้ต้องมี 2 มิติ โดย Iris มี 4 มิติ จะให้ user เลือก มา 2 column แรก สีเปลี่ยนตาม class เพื่อให้เห็น อิทธิพลของ class ก็ได้
