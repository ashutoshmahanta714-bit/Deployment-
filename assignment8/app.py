import numpy as np
import streamlit as st

mean=np.array([3.754071661237785,121.97231270358306,72.26058631921823,29.180781758957654,141.70521172638436,32.31482084690553,0.46725244299674273,33.24267100977199])
scale=np.array([3.3082329491254088,30.16616655707387,11.992146329894322,9.033139604547097,88.30742535873965,7.049355450274947,0.33842430884876556,11.555054594279584])
coef=np.array([0.35710897637594646,1.2014628775843823,-0.01639302440509654,0.06267038729376703,-0.04148983627804855,0.5980929745701028,0.400497012392302,0.19200958243822586])
intercept=-0.8820050733753169

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
    data=np.array([pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age])
    data=(data-mean)/scale
    probability=1/(1+np.exp(-(intercept+np.dot(data,coef))))
    prediction=1 if probability>=0.5 else 0
    if prediction==1:
        st.error(f'Diabetic - Probability: {probability:.2f}')
    else:
        st.success(f'Not Diabetic - Probability: {probability:.2f}')
