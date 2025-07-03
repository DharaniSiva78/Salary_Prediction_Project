import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))
model = joblib.load(MODEL_PATH)

st.title("Salary Prediction App")

input_df = pd.DataFrame([{
    "Total_Experience": st.slider("Total Experience", 0, 40, 5),
    "Total_Experience_in_field_applied": st.slider("Relevant Experience", 0, 40, 3),
    "Department": st.selectbox("Department", ["Engineering", "HR", "Sales", "Other"]),
    "Role": st.text_input("Role", "Software Developer"),
    "Industry": st.text_input("Industry", "IT"),
    "Organization": st.text_input("Organization", "ABC Corp"),
    "Designation": st.text_input("Designation", "Developer"),
    "Education": st.selectbox("Education", ["B.Tech", "M.Tech", "MBA", "Other"]),
    "Graduation_Specialization": st.text_input("UG Specialization", "Computer Science"),
    "University_Grad": st.text_input("UG University", "XYZ University"),
    "Passing_Year_Of_Graduation": st.number_input("Year of Graduation", min_value=2000, max_value=2030, value=2018),
    "PG_Specialization": st.text_input("PG Specialization", ""),
    "University_PG": st.text_input("PG University", ""),
    "Passing_Year_Of_PG": st.number_input("Year of PG", min_value=2000, max_value=2030, value=2000),
    "PHD_Specialization": st.text_input("PhD Specialization", ""),
    "University_PHD": st.text_input("PhD University", ""),
    "Passing_Year_Of_PHD": st.number_input("Year of PhD", min_value=2000, max_value=2030, value=2000),
    "Curent_Location": st.text_input("Current Location", "Bangalore"),
    "Preferred_location": st.text_input("Preferred Location", "Bangalore"),
    "Current_CTC": st.number_input("Current CTC", min_value=0, value=600000),
    "Inhand_Offer": st.number_input("Number of Offers", min_value=0, value=0),
    "Last_Appraisal_Rating": st.number_input("Appraisal Rating", min_value=0.0, max_value=5.0, value=3.5),
    "No_Of_Companies_worked": st.number_input("Companies Worked", min_value=0, value=2),
    "Number_of_Publications": st.number_input("Publications", min_value=0, value=0),
    "Certifications": st.number_input("Certifications", min_value=0, value=1),
    "International_degree_any": st.selectbox("International Degree", ["Yes", "No"])
}])

if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted CTC: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
