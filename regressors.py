import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression # มันจะไป map กับ data ที่เป็นเส้นตรง เพิ่มคือ จัดการ data จาก least-square ให้เป็ย polynomial
from sklearn.preprocessing import PolynomialFeatures

## โจทย์ non-linear ทำ regression => หาเส้นที่ track data ได้
x = np.random.rand(200)
y = 3 * np.sin(2 * x + 1) + np.random.normal(0, 0.1, len(x))

regressor = st.selectbox('select regressor', ['LR', 'KNN', 'SVM', 'Tree', 'Random Forest'])

if regressor == 'LR': # ต้องเปลี่ยน สมการ X ใหม่
    degree = st.select_slider('select degree', options=range(1, 11))
    poly = PolynomialFeatures(degree=degree)
    x_ = poly.fit_transform(x[:, None])
    rgs = LinearRegression()

if regressor == 'KNN':
    k = st.select_slider('select k', options=range(1, 11))
    rgs = KNeighborsRegressor(n_neighbors=k)

if regressor == 'SVM':
    C = st.select_slider('select C', options=range(1, 11))
    kernel = st.selectbox('select kernel', ['linear', 'poly', 'rbf'])
    rgs = SVR(C=C, kernel=kernel)

if regressor == 'Tree':
    depth = st.selectbox('select depth', options=range(1, 10))
    rgs = DecisionTreeRegressor(max_depth=depth)

if regressor == 'Random Forest':
    n_estimators = st.selectbox('select n_estimators', options=range(1, 10))
    rgs = RandomForestRegressor(n_estimators=n_estimators)

if regressor == 'LR':
    rgs.fit(x_, y)
    z = rgs.predict(x_)
else:
    rgs.fit(x[:, None], y) # ต้องการ 100,1 แต่มี 200 ก็ใส่ none # Expected 2D array, got 1D array instead: when run this "rgs.fit(x, y)"
    z = rgs.predict(x[:, None])

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, z, '.r')

st.pyplot(plt)