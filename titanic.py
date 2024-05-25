import seaborn as sns
import streamlit as st
import numpy as np
# -----
import torch
from torch import nn


# [4] 128 [3]
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 128)
        # self.fc2 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(128,
                             1)  # เราไม่นิยม ถ้า มี 2 class เราจะใช้เป็น 1 แทน (dummy variable problem) # เพราะยังไง เราก็รู้ อีกอัน และเป็นแบบนี้จะใช่ softmax ไม่ได้ เพราะมันดูหลาย class ทำให้ตัว loss ใช้ CrossEntropyloss ไม่ได้ (เพราะเหมาะกับ3 ขึ่นไป) ซึ่ง 0-1 มันเป็น float ก็ต้อง ทำตัว train, test ให้เป็น type เดียวกัน

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = nn.functional.leaky_relu(self.fc2(x))
        return torch.sigmoid(self.fc2(x))  # sigmoid rank ค่า 0-1

model = Classifier()
model.load_state_dict(torch.load("model_titanic_cls.pth"))
model.eval()
# -----

st.title('Titanic Dataset')
st.sidebar.title('Input Parameters')

# Collecting input parameters from the user
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], help="Select the passenger class.", format_func=lambda x: '1st' if x == 1 else '2nd' if x == 2 else '3rd')
age = st.sidebar.number_input("Age", min_value=1, value=1, help="Enter the age of the passenger. Minimum value is 1.")
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, value=0, help="Enter the number of siblings/spouses aboard.")
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, value=0, help="Enter the number of parents/children aboard.")
fare = st.sidebar.number_input("Fare", min_value=0.0, value=0.0, help="Enter the fare paid by the passenger.")
sex = st.sidebar.selectbox("Sex", ["male", "female"], help="Select the sex of the passenger.")
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"], help="Select the embarkation port.")
who = st.sidebar.selectbox("Who", ["child", "man", "woman"], help="Select the passenger type (child, man, woman).")

# Convert categorical variables to one-hot encoding
sex_male = 1 if sex == "male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
who_child = 1 if who == "child" else 0
who_man = 1 if who == "man" else 0
who_woman = 1 if who == "woman" else 0

# Create the input feature vector (12 elements)
x = [
    pclass, age, sibsp, parch, fare,
    sex_male, embarked_C, embarked_Q, embarked_S,
    who_child, who_man, who_woman
]

st.write([x])
st.write(type[x])

# Converting to a tensor
x = torch.tensor([x], dtype=torch.float32) # ตรงนี้ ต้องเอา [] ครอบ เพื่อให้ tensor มิติ 1 x 13

labels = ['survive', 'dead']
with torch.no_grad():
    y_ = model(x)
    # Adjusted the prediction logic to use a threshold of 0.5 for binary classification.
    # This line dynamically computes the prediction based on the input parameters
    prediction = (y_ >= 0.5).item()

st.write('Predicted survival:', labels[prediction])