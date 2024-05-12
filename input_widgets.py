import streamlit as st
from PIL import Image, ImageColor
import numpy as np
import pandas as pd

name = st.text_input('Enter your name')
st.write('Hello', name)
age = st.number_input('Enter your age', step=1,
                      min_value=0, max_value=130)
st.write('Your age is', age)
bio = st.text_area('Enter your bio', height=200)
birthday = st.date_input('Enter your birthday')
st.write(birthday)
time = st.time_input('What time did you wake up?')
st.write(time)
avatar = st.camera_input('Cheese!!')

# แก้ error ที่คล้ายๆ แบบนี้ส่วนใหญ่ใช้ ท่านี้
if avatar is not None:
    avatar = np.array(Image.open(avatar))[::-1, :, :]
    st.image(avatar) # กดก่อน ถึงจะ debug ได้
    # ควร เก็บค่าไว้ใน memory ( RAM ) |  Byte IO คือ ประพฤตตัวคล้ายกับ เขียนค่าลง hdd แต่ เราไม่ต้องการให้เขียน read/write บ่อยๆ เพราะมันจะเกิดที่เครื่องเราเยอะ
    # ดังนี้ ก็เก็บไว้ใน RAM และก็
    Image.fromarray(avatar).save('avatar_server.png') # ลงhdd
    st.download_button('Download image',
                       open('avatar_server.png', 'rb'),
                       'avatar_client.png')

st.write('server image')

# upload files
img = st.file_uploader('Upload image', type=['jpg', 'png'])
if img is not None:
    st.image(Image.open(img))

clicked = st.button('Click me!')
if clicked:
    st.write('Button clicked!')

checked = st.checkbox('Check me!')
if checked:
    st.write('Checkbox checked!')

choice = st.radio('Choose one', ['A', 'B', 'C'])
st.write('You chose', choice)

mul_choices = st.multiselect('Choose many', ['A', 'B', 'C'])
st.write('You chose', mul_choices)

rating = st.slider('Rate me', 1, 5, 1)
st.write('You rated', rating)

level = st.select_slider('Select level', options=['Beginner', 'Intermediate', 'Advanced'])
st.write('You selected', level)

color = st.color_picker('Pick a color')
st.write(f'You picked {ImageColor.getcolor(color, "RGB")}')

## โจทย์ upload csv -> show df

