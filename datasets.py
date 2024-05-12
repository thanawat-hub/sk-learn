import streamlit as st
import seaborn as sns

st.write("Datasets")

ds = st.selectbox("Select Dataset",
             sns.get_dataset_names())# ให้ user เลือก dataset ซึ่งจะต้องมี widget ie. st.selectbox etc

df = sns.load_dataset(ds)

# display
# st.dataframe(df) # medthod 1
df # echo varible มันจะ เอา st.write มาหุ้มให้เลย medthod 2 เรียกว่า magic