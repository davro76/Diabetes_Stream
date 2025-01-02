#!/usr/bin/env python
# coding: utf-8

# # Librairies and dependencies

# In[270]:


# import appropriate libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report #predict_classes
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# # Dataset 

# In[272]:


# read the dataset
diabetes = pd.read_csv(r"C:\Users\daver\OneDrive\Desktop\diabetes_prediction\diabetes.csv")

# display the first 2 rows of the dataset
diabetes.head(2)


# # Data Processing 

# ### Feature titles homogenization  

# In[276]:


# replace space with underscore use lowercase for column names
diabetes.columns=diabetes.columns.str.replace(' ','_').str.lower()
diabetes_df =diabetes
diabetes_df.head(2)


# ### Dataset size

# In[279]:


# shape of the dataframe

diabetes_size=diabetes_df.shape
lines,columns = diabetes_df.shape
print(f"The shape of the dataframe is the following: {diabetes_size}")
print("The dataframe as a total number of line of:", lines)
print("The dataframe as a total number of column of:", columns)


# ### Features data types

# In[282]:


# features and data type
diabetes_df.info()


# ### Missing Data

# In[285]:


# handle missing data
diabetes_df.isnull().sum() 


# ### Outliers

# In[288]:


# detect possible outlier using boxplot
fig = px.box(diabetes_df,y="age",
title='Original boxplot for age')
fig.show()


# ## No need to remove the outliers.

# # Eploratory Data Analysis 

# In[292]:


#plt.style.available


# In[294]:


# distribution of the continous variable age
fig=px.histogram(diabetes_df,x="age",
title='Distribution for age')
fig.show()


# #### The variable age is normally distributed, and most of the data is collected from patients aged between 24 and 73

# In[297]:


bar_df = diabetes_df.groupby(['class','gender']).size().reset_index(name='count')
fig=px.bar(bar_df,y='count', x='gender',color='class',orientation='v',barmode='group', title='Count of class patient by gender')
fig.show()


# In[256]:


# class distribution
fig=px.pie(diabetes_df,names='class',title='Distribution of the class feature')
fig.show()


# ### Based on the above pie chart, the dataset is imbalanced. Therefore, we need to use resampling before feeding this dataset into the MLP model.

# In[264]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Example dataset
data = {
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'class': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive'],
    'polyuria': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()  # Ensure scalar sum
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))  # Non-negative
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Create a correlation matrix for categorical variables
categorical_columns = ['gender', 'class', 'polyuria']
correlation_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            correlation_matrix.loc[col1, col2] = 1.0
        else:
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            correlation_matrix.loc[col1, col2] = cramers_v(confusion_matrix)

# Convert to numeric for heatmap plotting
correlation_matrix = correlation_matrix.astype(float)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
plt.title("Cramér's V Heatmap of Categorical Variables")
plt.show()


# ### Features and target

# In[43]:


# create features and target
X= diabetes_df.drop('class',axis=1) #features

Y= diabetes_df['class'] # target
print(X.shape)
print(Y.shape)


# ### Convert text features to numerical values

# In[82]:


# text features and target to numeric values

enc = LabelEncoder() # instance of LabelEncoder
cols = X.select_dtypes(include=['object']).columns #grab only the categorical data
Y = enc.fit_transform(Y.astype('str'))

# for loop to iterate through each feature
for col in cols:
    X[col] = enc.fit_transform( X[col].astype('str'))
X.head(2)


# ### Check Class Imbalance
# A dataset is considered imbalanced if the minority class makes up less than 40%<br> (or sometimes 30%,depending on the domain) of the total samples.

# In[128]:


# compute and display the percentage for each class category
#if round(Y.value_counts(1)*100,2)['Positive'] == round(Y.value_counts(1)*100,2)['Negative']:
   # print('Balanced data set')
#elif round(Y.value_counts(1)*100,2)['Positive'] < 40.00:
   # print(round(Y.value_counts(1)*100,2)['Positive'])
   # print('Imbalanced data set')
#elif round(Y.value_counts(1)*100,2)['Negative'] < 40.00:
   # print(round(Y.value_counts(1)*100,2)['Negative'])
   # print('Imbalanced dataset')


# ### Split the dataset between train, validation, and test data

# In[86]:


# first split for train and test data
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2) # 80% train and 20% test 


# In[88]:


X_train.shape


# ### Resample the train dataset (features and target) using SMOTE algorithm

# In[91]:


sme = SMOTE(random_state=42) # create a smote instance
X_resampled, y_resampled = sme.fit_resample(X_train, y_train)  # resample the train dataset


# # Let's Build the Neural Network Model

# In[152]:


# Define the model
model = Sequential()

# Input layer
model.add(Input(shape=(16,)))  

# Hidden layers
model.add(Dense(64, activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(16, activation='relu'))  # Third hidden layer

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()


# In[154]:


# Train the model for 200 epochs
model.fit(X_resampled, y_resampled, epochs=200)


# # Results Interpretation

# ## Testing the model

# In[156]:


# Evaluate the model on the training data
scores = model.evaluate(X_train, y_train, verbose=0) # verbose=0, Suppresses unnecessary output during evaluation.

# Print the training accuracy
training_accuracy = scores[1] * 100
print(f"Training Accuracy: {training_accuracy:.2f}%")

# Evaluate the model on the testing data
scores = model.evaluate(X_test, y_test, verbose=0) # verbose=0, Suppresses unnecessary output during evaluation.

# Print the testing accuracy
testing_accuracy = scores[1] * 100
print(f"Testing Accuracy: {testing_accuracy:.2f}%")


# In[170]:


# Use model.predict() to get predicted probabilities
predicted_probabilities = model.predict(X_test)

# Convert probabilities to class predictions (threshold of 0.5 for binary classification)
y_test_pred = (predicted_probabilities > 0.5).astype(int).flatten()

# Generate the confusion matrix
c_matrix = confusion_matrix(y_test, y_test_pred)

# Create combined labels with numeric values and descriptive text
numeric_values = c_matrix.astype(str)
text_labels = np.array([['True Negative', 'False Negative'], 
                        ['False Positive', 'True Positive']])
combined_labels = np.array([[f"{numeric_values[i][j]}\n{text_labels[i][j]}" for j in range(2)] for i in range(2)])

# Plot the confusion matrix using seaborn
ax = sns.heatmap(c_matrix, annot=combined_labels, fmt='',
                 xticklabels=['No Diabetes', 'Diabetes'],
                 yticklabels=['No Diabetes', 'Diabetes'],
                 cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")


# In[266]:


import streamlit as st

# Import necessary modules
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
MODEL_PATH = 'mlp_diabetes_model.pkl'
SCALER_PATH = 'scaler.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define navigation pages
PAGES = {
    "Home": "home",
    "Predict Diabetes": "predict",
    "About": "about",
}

# Home Page
def home_page():
    st.title("Welcome to the Diabetes Prediction App")
    st.write(
        """
        This application uses a pre-trained MLP model to predict whether a person
        is likely to have diabetes based on input features. Use the navigation menu to proceed.
        """
    )

# Prediction Page
def prediction_page():
    st.title("Diabetes Prediction")

    # Collect user input
    st.write("Enter the following details to predict diabetes:")
    pregnancies = st.number_input("Number of Pregnancies:", min_value=0, step=1)
    glucose = st.number_input("Glucose Level:", min_value=0.0)
    blood_pressure = st.number_input("Blood Pressure:", min_value=0.0)
    skin_thickness = st.number_input("Skin Thickness:", min_value=0.0)
    insulin = st.number_input("Insulin Level:", min_value=0.0)
    bmi = st.number_input("BMI:", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0)
    age = st.number_input("Age:", min_value=0, step=1)

    # Make a prediction when the button is clicked
    if st.button("Predict"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        if prediction[0] == 1:
            st.warning(f"The model predicts that you may have diabetes. Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.success(f"The model predicts that you do not have diabetes. Probability: {prediction_proba[0][0]:.2f}")

# About Page
def about_page():
    st.title("About This App")
    st.write(
        """
        This app is built using Streamlit and a trained Multi-Layer Perceptron (MLP) model.
        The model was trained using a dataset to predict diabetes based on health metrics.
        """
    )

# Main Functionality
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        home_page()
    elif selection == "Predict Diabetes":
        prediction_page()
    elif selection == "About":
        about_page()

if __name__ == "__main__":
    main()


# In[ ]:


# import streamlit lybrary
import streamlit as st
st.write('Diabetes Predictor App')

