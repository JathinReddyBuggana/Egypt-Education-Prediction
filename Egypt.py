import pandas as pd
import pickle
import streamlit as st

# Load the pre-trained model
model = pickle.load(open(r"best_logistic_regression_model.pkl","rb"))

# Streamlit App
st.title("Egypt Education Dataset Prediction")

# User Input Features
st.header("Input Features")

# User input fields for each feature
student_age = st.number_input("Student Age", min_value=1, max_value=100)
student_year = st.selectbox("Student year", ["1", "2", "3"]) 
text1 = {1:"None", 2:"High School", 3:"College"}
father_degree = st.selectbox("Father Degree", text1.keys())  # Example categories
st.markdown(f"<h3 style='font-size:20px;'>Father Degree: {text1[father_degree]}</h3>",unsafe_allow_html=True)
text2 = {1:"None", 2:"High School", 3:"College"}
mother_degree = st.selectbox("Mother Degree", text2.keys())  # Example categories # Ensure the feature matches training data
st.markdown(f"<h3 style='font-size:20px;'>Mother Degree: {text2[mother_degree]}</h3>",unsafe_allow_html=True)

# Input for subjects
subjects = []
for i in range(1, 11):
    subjects.append(st.number_input(f"Subject {i}", min_value=0.0, max_value=100.0))

# Function to rename features to match what the model expects
def rename_features(input_data):
    feature_mapping = {
        'Student Age': 'Student Age',  # Exact match  # Adjust to whatever your model expects
        'Student Year': 'Student year',
        'Father Degree': 'Father Degree',
        'Mother Degree': 'Mother Degree',  # Change to match expected name
        'Subject_1': 'Subject_1',
        'Subject_2': 'Subject_2',
        'Subject_3': 'Subject_3',
        'Subject_4': 'Subject_4',
        'Subject_5': 'Subject_5',
        'Subject_6': 'Subject_6',
        'Subject_7': 'Subject_7',
        'Subject_8': 'Subject_8',
        'Subject_9': 'Subject_9',
        'Subject_10': 'Subject_10',
    }
    return input_data.rename(columns=feature_mapping)

# Make prediction when the button is clicked
if st.button("Predict"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'Student Age': [student_age],
        'Student year': [student_year], 
        'Father Degree': [father_degree],  
        'Mother Degree': [mother_degree],   
        'Subject_1': [subjects[0]],
        'Subject_2': [subjects[1]],
        'Subject_3': [subjects[2]],
        'Subject_4': [subjects[3]],
        'Subject_5': [subjects[4]],
        'Subject_6': [subjects[5]],
        'Subject_7': [subjects[6]],
        'Subject_8': [subjects[7]],
        'Subject_9': [subjects[8]],
        'Subject_10': [subjects[9]],
    })

    # Log the input data for debugging
    st.write("Input Data before renaming:", input_data)

    # Rename features to match the model's expected feature names
    input_data = rename_features(input_data)
    st.write("Input Data after renaming:", input_data)
    
    #prediction = model.predict(input)
    #st.write(f"Predicition class:{prediction[0]}")



    try:
        #Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted class: {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")





