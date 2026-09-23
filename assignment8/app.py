import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('diabetes.csv')
zero_col=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[zero_col]=df[zero_col].replace(0,np.nan)
df.fillna({'Glucose':df.Glucose.median(),
           'BloodPressure':df.BloodPressure.median(),
           'SkinThickness':df.SkinThickness.median(),
           'Insulin':df.Insulin.median(),
           'BMI':df.BMI.median()},inplace=True)

x=df.drop(columns='Outcome')
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)

lr=LogisticRegression()
lr.fit(x_train,y_train)

st.title('Diabetes Prediction')
st.write('Enter the patient details to predict diabetes.')

pregnancies=st.number_input('Pregnancies',min_value=0,value=1)
glucose=st.number_input('Glucose',min_value=0,value=120)
blood_pressure=st.number_input('Blood Pressure',min_value=0,value=70)
skin_thickness=st.number_input('Skin Thickness',min_value=0,value=20)
insulin=st.number_input('Insulin',min_value=0,value=80)
bmi=st.number_input('BMI',min_value=0.0,value=25.0)
diabetes_pedigree_function=st.number_input('Diabetes Pedigree Function',min_value=0.0,value=0.5)
age=st.number_input('Age',min_value=1,value=30)

if st.button('Predict'):
    data=[[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]]
    data=sc.transform(data)
    prediction=lr.predict(data)
    probability=lr.predict_proba(data)[:,1]
    if prediction[0]==1:
        st.error(f'Diabetic - Probability: {probability[0]:.2f}')
    else:
        st.success(f'Not Diabetic - Probability: {probability[0]:.2f}')
