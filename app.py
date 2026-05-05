import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Student GPA Predictor")

age = st.number_input("Age", min_value=15, max_value=18, value=16, step=1)
study = st.number_input("Study Time Weekly", min_value=0.0, max_value=20.0, value=10.0, step=0.5)
absences = st.number_input("Absences", min_value=0, max_value=29, value=0, step=1)

if st.button("Predict GPA"):
    sample = pd.DataFrame([[age,0,0,0,study,absences,0,0,0,0,0,0]],
    columns=["Age","Gender","Ethnicity","ParentalEducation","StudyTimeWeekly",
             "Absences","Tutoring","ParentalSupport","Extracurricular",
             "Sports","Music","Volunteering"])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    st.write("Predicted GPA:", prediction[0])
