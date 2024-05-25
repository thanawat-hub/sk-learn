import streamlit as st

from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, SelectPercentile, GenericUnivariateSelect, \
                                      chi2,  r_regression, f_regression, mutual_info_regression
import pandas as pd
import altair as alt # to plot

X = load_diabetes(as_frame=True)

# Sidebar section for user input
st.sidebar.header("Feature Selection Parameters")
score_func = st.sidebar.selectbox("Score function",
                                  ["r_regression", "f_regression", "mutual_info_regression"])

mode = st.sidebar.selectbox("Selection Mode", ["k_best", "percentile"]) #ที่เหลือนี้ไปใช้กับ classif , "fpr", "fdr", "fwe"]) เพราะบางทีค่าอาจจะหาย
if mode == 'k_best':
    param = st.sidebar.slider("Number of feature", 1, 10, 5)

if mode == 'percentile':
    param = st.sidebar.slider("Percentile", 1, 100, 50)

if mode == ['fpr', 'fdr', 'fwe']:
    param = st.slider.slider("Alpha", 0.01, 0.1, 0.05)

# eval ทำหน้าที่ เปลี่ยน string ให้เป็น statement code ทำให้ exceute string ได้เลย
selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
selector.fit(X.data, X.target)

df = pd.DataFrame({"feature": X.data.columns, "score": selector.scores_})
df_sorted = df.sort_values(by='score').reset_index(drop=True) # sort จากชื่อ column หรือจะบน index ก็ได้, inplace ไม่ต้องเอาตัวแปรมารับ
# sorting ก่อน ทำตรงนี้ X = df ที่มี ค่า output คู่กับ Y = column ทั้งหมดของ X

# ใน alt มันจะพยายาม sort ให้เอง แล้วไม่ได้ ต้องมาท่าประมาณนี้ คือไป sort มาเองก่อน แล้วก็บอกมันไม่ต้อง sort นะจ๊ะ
chart = alt.Chart(df_sorted).mark_bar().encode(
    x=alt.X('feature', sort=None),  # Ensure that sorting is preserved as per DataFrame
    y=alt.Y('score', sort=None) # Use sort=None to ensure the chart uses the order from the DataFrame
).properties(
    title='Feature Scores'
)

col1, col2 = st.columns(2)
with col1:
    # Display the chart
    st.altair_chart(chart)
with col2:
    # จะได้ feature importance เรียงตาม score ในของแต่ละ score_func
    st.write("Selected feature:",selector.get_feature_names_out())