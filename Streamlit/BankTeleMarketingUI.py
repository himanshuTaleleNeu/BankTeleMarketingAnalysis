import streamlit as st
from BankTeleMarketing import predict_response

st.set_page_config(
    page_title="BankTeleMarketing UI",
    page_icon="🏦"
)
st.header("Bank Tele Marketing Campaign")

# AGE
age = st.text_input("Enter your age", "")

if age:
    age = int(age)  
    if age > 120:
        st.error("Your age is greater than 120. Please enter a valid age.")
    else:
        st.success(f"Your age: {age}")

# DEFAULT
# default = st.selectbox("Select the default value", [0, 1], index=None)
# st.write(f"Selected value: {default}")

default_options = ["Please choose one option", 0, 1]
default = st.selectbox("Default", default_options)

if default == "Please choose one option":
    st.write("Default: N/A")
else:
    st.write(f"Default: {default}")

# Balance
try:
    balance = st.number_input("Enter Balance", min_value=0)
except ValueError:
    st.error("Please enter a valid numerical value for the Balance.")
if isinstance(balance, float):
    st.write(f"Balance Value: {balance}")
else:
    st.write("Balance Value: N/A")

# Housing
housing_options = ["Please choose one option", 0, 1]
housing = st.selectbox("Housing", housing_options)

if housing == "Please choose one option":
    st.write("Housing: N/A")
else:
    st.write(f"Housing: {housing}")

# Loan
loan = st.selectbox("Select the loan ", [0, 1], index=None)
st.write(f"Selected value: {loan}")

# JOB
job_categories = ["Choose one option", "admin", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
                 "self-employed", "services", "student", "technician", "unemployed", "unknown"]
job = st.selectbox("Select a job category", job_categories)

if job == "Choose one option":
    st.write("Job Category: N/A")
else:
    st.write(f"Job Category: {job}")

# Education
education_categories = ["Choose one option", "primary", "secondary", "tertiary", "unknown"]
education = st.selectbox("Select an Education category", education_categories)

if education == "Choose one option":
    st.write("Education Category: N/A")
else:
    st.write(f"Education Category: {education}")

if st.button("Predict"):
    if (
         age and
         default != "Please choose one option" and
         balance is not None and
         housing != "Please choose one option" and
         loan != "Please choose one option"
        ):

        # Convert user inputs and call the predict_response function from model_prediction.py
        response = predict_response(age, default, balance, housing, loan, job, education)
        # Display the prediction result
        st.success(f"Predicted Response: {response}")
    else:
        st.warning("Please fill in all the input fields before predicting.")


