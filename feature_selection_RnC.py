#same each file "feature_selection_classification.py", feature_selection_regression.py" but in one file seperate with tap

import streamlit as st
from sklearn.datasets import load_diabetes, load_iris
import altair as alt
from sklearn.feature_selection import (GenericUnivariateSelect,
                                       r_regression, f_regression, mutual_info_regression,
                                       chi2, f_classif, mutual_info_classif)
import pandas as pd
tab1, tab2 = st.tabs(['Regression', 'Classification'])
mode = st.sidebar.selectbox("Mode", ['k_best', 'percentile'])


with tab1:
    st.subheader("Feature selection for Regression")
    X = load_diabetes(as_frame=True)
    if mode == 'k_best':
        max_feature = X.data.shape[1]
        param = st.sidebar.slider("Number of features", 1, 10, int(max_feature/2))
    if mode == 'percentile':
        param = st.sidebar.slider("Percentile", 1, 100, 50)

    score_func = st.selectbox("Score function",
                              ['r_regression', 'f_regression', 'mutual_info_regression'],
                              key='regression')
    selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
    selector.fit(X.data, X.target)
    df = pd.DataFrame({'feature': X.data.columns, 'score': selector.scores_})
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('feature', sort='-y'),
        alt.Y('score')
    )
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart)
    with col2:
        st.write("Selected features:", selector.get_feature_names_out())

with tab2:
    st.subheader("Feature selection for Classification")
    X = load_iris(as_frame=True)
    if mode == 'k_best':
        max_feature = X.data.shape[1]
        param = st.sidebar.slider("Number of features", 1, 10, int(max_feature/2))
    if mode == 'percentile':
        param = st.sidebar.slider("Percentile", 1, 100, 50)
    score_func = st.selectbox("Score function",
                              ['chi2', 'f_classif', 'mutual_info_classif'],
                              key='classification')
    selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
    selector.fit(X.data, X.target)
    df = pd.DataFrame({'feature': X.data.columns, 'score': selector.scores_})
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('feature', sort='-y'),
        alt.Y('score')
    )
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart)
    with col2:
        st.write("Selected features:", selector.get_feature_names_out())