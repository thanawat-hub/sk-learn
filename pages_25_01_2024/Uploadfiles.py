import streamlit as st
import pandas as pd

st.title("Datasets")
# upload files
file = st.file_uploader('Upload data support csv', type=['.csv', '.pdf'])
if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df)
