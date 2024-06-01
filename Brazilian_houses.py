# index ไฟล์นี้ เอาไว้สำหรับ load data
# ปกติใน streamlit มันรันจากบนลงล่าง ทำให้เปลืองทรัยากร วิธีแก้คือ
# 1. cache cross-session เข้าถึง X ได้พร้อมกัน เช่น datasets (ค่าเหมือนกันหมด) ก็เก็บที่ level นี้ วิธีคือ ผ่าน decorator @st.cache แล้วไปดูว่าจะใช้ อะไร กับ session อะไร
# 2. session (single session) มันอิสระต่อกัน เช่นไปดูค่าเดียวกัน session1 (มองเป็น user) ไว้ปรับแต่งของตัวเอง session2 จะไม่เห็นของ session1
# impute ก่อน
import numpy as np
import pandas as pd

import streamlit
import streamlit as st

from sklearn.impute import KNNImputer

df = pd.read_csv('houses_to_rent_v2.csv')

# backend ทำงาน cache บน memory ตรงนี้คือ level cache
@streamlit.cache_data
def load_data():
    # return pd.read_csv('houses_to_rent_v2.csv')

    # ใส่ impute ให้ทุกคนก่อน | เราทำ POC บน ipynb แล้ว
    df = pd.read_csv('houses_to_rent_v2.csv')
    imputer = KNNImputer(n_neighbors=3)
    df.floor = df.floor.replace('-', np.nan).astype(float)
    numeric_col = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df[numeric_col] = imputer.fit_transform(df[numeric_col])
    return df


# ตรงนี้คือ session level
# คือหลังบ้าน ทำการโหลด df มารอไว้ละ
if 'df' not in st.session_state:
    # st.set_page_config(['df']) = load_data()
    with st.spinner('Loading data...'):
        st.session_state['df'] = load_data()

st.title('Brazilian Houses')
st.header('Rent price prediction')
st.image("https://qph.cf2.quoracdn.net/main-qimg-8b7db09f0026709117d0369aeaaee360-lq")