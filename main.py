import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from joblib import load
st.header("Logistic regression Model")
data=load_iris()
st.subheader("this is the dataset")
st.write(pd.DataFrame(data.data,columns=data.feature_names))


code="""from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Using StandardScaler -----
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

model_std = LogisticRegression(max_iter=10000)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)

print("Accuracy with StandardScaler:", accuracy_score(y_test, y_pred_std))


# ----- Using MinMaxScaler -----
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

model_mm = LogisticRegression(max_iter=10000)
model_mm.fit(X_train_mm, y_train)
y_pred_mm = model_mm.predict(X_test_mm)

print("Accuracy with MinMaxScaler:", accuracy_score(y_test, y_pred_mm))

        """
st.code(code,language="python")

result="""Accuracy with StandardScaler: 0.9736842105263158
Accuracy with MinMaxScaler: 0.9824561403508771"""
st.code(result,language="python")

st.write("based on this i have chosen minmax")

model=load('C:/Users/kaush/Desktop/End-End-Ml/model.joblib')
if model:
    st.header("*Model is online*")
    with st.form("my_form"):
        st.write("Inside the form")
        input=st.text_input("enter the value sep by ',")
        st.form_submit_button("check")
        # input=list(map(int,st.text_input("enter the value sep by ',").split(',')))
    trim=lambda x:x.strip()
    input =list(map(trim,input.split(',')))
    input=list(map(float,input))
    predict=model.predict([input])
    if predict==0:
        st.subheader("no cancer")
    else:
        st.subheader(" cancer")
else:
    st.header("*Model is offline*")
        