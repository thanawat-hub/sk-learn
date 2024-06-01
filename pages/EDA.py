import streamlit as st
import pandas as pd
import plotly.express as px# ได้ html

# ดูภาพรวม, เจาะทีละ feature
st.title('EDA')#show data, graph
if 'df' in st.session_state:#จะเริ่มทำ ก็ต่อเมื่อ load data แล้ว
    df = st.session_state['df']#หยิบมาใส่ในตัวแปรdf

    selected_col = st.sidebar.multiselect('Select columns', df.columns)
    if len(selected_col) == 0:#ถ้าไม่เลือกเลย
        selected_col = df.columns#ก็เอาไปเลยทุกอัน
    st.write(df[selected_col])

    # อาจจะ design ว่า ถ้าเลือก column ไหน ก็เอามา plot corr แต่ทำได้แค่ numeric
    corr = df[selected_col].corr(numeric_only=True).round(2)#2ตำแหน่ง
    fig = px.imshow(corr, text_auto=True)#plot heat map
    # fig.show()#ถ้าใส่มันจะเด้งไป port ของ plotly
    st.plotly_chart(fig)#ต้องเรียกใช้บน st เนอะ ว่าใช้ interactive ตัวไหน


    col = st.sidebar.selectbox('Select columns', df.columns)
    tmp = df[col]
    if pd.api.types.is_numeric_dtype(tmp):
        outliers = st.sidebar.checkbox('Outliers', False)#สำหรับตัวที่เป็นเลข ก็ควรมีการตัด outliers
        if outliers:
            q_low = tmp.quantile(0.01)
            q_high = tmp.quantile(0.99)
            tmp = tmp[(tmp > q_low) & (tmp < q_high)]
        st.write(tmp.describe())
        fig = px.histogram(tmp, x=col)
        st.plotly_chart(fig)

    else:#เพราะไม่ใช่ตัวเลข
        st.write(tmp.value_counts())
        fig = px.pie(tmp, names=col)
        st.plotly_chart(fig)
        #plot pie chart ว่ากินพื้นที่เท่าไหร่


    # ถ้าค่ามันไปกระจุกอยู่ที่หนึ่งก็ให้ กำจัด outliner ออกไป
    # ถ้าเกิน ช่วงนั้นตัดทิ้ง