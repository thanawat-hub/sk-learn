import streamlit as st
import pandas as pd

# UCI Iris dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Read the dataset into a pandas DataFrame
iris_df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# Display the first few rows of the dataset
print(iris_df.head())

st.title("ML with Iris")
st.write("This is Iris dataset from UCI")
st.dataframe(iris_df)

