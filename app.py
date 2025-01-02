# install useful librari
import streamlit as st
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

diabetes = pd.read_csv(r"C:\Users\daver\OneDrive\Desktop\diabetes_prediction\diabetes.csv")

# Define navigation pages
PAGES = {
    "Home": "home",
    "Predict Diabetes": "predict",
    "Dataset": "dataset",
}

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Main body background */
        .stApp {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important;
            color: #333333 !important;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background: green !important;
            color: white !important;
        }

        /* Top bar customization */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #ffcc00, #ff9900) !important;
            color: white !important;
            text-align: center;
        }

        /* Center header title */
        h1 {
            text-align: center;
            color: #333 !important;
        }

        /* Customize widget labels */
        div[data-testid="stMarkdownContainer"] > p {
            color: black !important;  /* Text color set to black */
        }

        /* Customize buttons and captions */
        div.stButton > button {
            background-color: #007bff !important;
            color: white !important;
            border-radius: 5px !important;
        }

        /* Center images in home page */
        .slider-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def home_page():
    st.title("Welcome to the Diabetes Prediction App")
    st.write(
        """
        This web application is designed as an educational simulation to predict the likelihood of diabetes in patients using a Multi-Layer Perceptron (MLP) neural network algorithm.\n\n The application utilizes a diabetes dataset comprising various features such as age, gender,\n\n and medical indicators like polyuria, polydipsia, sudden weight loss, weakness, and obesity, among others.\n\n The purpose of this app is purely informational and does not aim to replace professional medical diagnosis by a physician.\n\n Built with the Streamlit Python framework, the application is hosted on Streamlit Community Cloud\n\n and includes four interactive pages: the Home page, which introduces the app and its creator; the Data Processing page;\n\n an Exploratory Data Analysis (EDA) page; and a Prediction page where users can input features to simulate diabetes predictions.\n\n The app's source code is openly available on GitHub, promoting transparency and collaboration.
        """
    )

    # List of images to display
    images = [
        {
            "path": "pics/Diabetes_illustration.png",
            "caption": "Diabetes Illustration",
        },
        {
            "path": "pics/diabetes_counselling.jpg",
            "caption": "Diabetes Counselling",
        },
        {
            "path": "pics/diabetes_type.jpg",
            "caption": "Diabetes Type",
        },
    ]

    # Start an infinite loop to display images with a 7-second delay.
    for idx in range(len(images)):
        # Display the image one by one
        st.image(images[idx]["path"], caption=images[idx]["caption"], use_container_width=True)
        time.sleep(7)  # Wait for 7 seconds before changing the image

def prediction_page():
    st.title("Diabetes Prediction")
    st.write("Enter the following details to predict diabetes:")

    pregnancies = st.number_input("Number of Pregnancies:", min_value=0, step=1)
    glucose = st.number_input("Glucose Level:", min_value=0.0)
    blood_pressure = st.number_input("Blood Pressure:", min_value=0.0)
    skin_thickness = st.number_input("Skin Thickness:", min_value=0.0)
    insulin = st.number_input("Insulin Level:", min_value=0.0)
    bmi = st.number_input("BMI:", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0)
    age = st.number_input("Age:", min_value=0, step=1)

    if st.button("Predict"):
        # Placeholder for prediction logic
        st.success("Prediction logic goes here!")

        # Example: Display the entered data
        st.write(f"Pregnancies: {pregnancies}")
        st.write(f"Glucose Level: {glucose}")
        st.write(f"Blood Pressure: {blood_pressure}")
        st.write(f"Skin Thickness: {skin_thickness}")
        st.write(f"Insulin Level: {insulin}")
        st.write(f"BMI: {bmi}")
        st.write(f"Diabetes Pedigree Function: {dpf}")
        st.write(f"Age: {age}")

def dataset_page():
    st.title("About The Data")
    st.write(
        """
        This app is built using Streamlit and a trained Multi-Layer Perceptron (MLP) model.
        The model was trained using a dataset to predict diabetes based on health metrics.
        """
    )
    st.table(diabetes.head(7))
def main():
    # Apply custom CSS
    add_custom_css()
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        home_page()
    elif selection == "Predict Diabetes":
        prediction_page()
    elif selection == "Dataset":
        dataset_page()

if __name__ == "__main__":
    main()
