import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Title for the Streamlit app
st.title("Random Forest Classifier for Bird Type Classification")

# Load the data
data = pd.read_csv('data_csv1.csv')
st.write("Dataset Head:")
st.dataframe(data.head())

# Filter the data
dataset = data[data['bird_type'].isin(['astfly', 'bulori', 'warvir', 'woothr'])].drop(['filename'], axis=1)
st.write("Filtered Dataset Shape:", dataset.shape)

# Encode the labels
y = LabelEncoder().fit_transform(dataset.iloc[:, -1])
st.write("Encoded Labels:", y)

# Standardize the features
scaler = StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype=float))
x = scaler.transform(np.array(dataset.iloc[:, :-1], dtype=float))
st.write("Standardized Features Shape:", x.shape)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=52)
st.write('Train set shape:', x_train.shape, y_train.shape)
st.write('Test set shape:', x_test.shape, y_test.shape)

# Train the model
model = RandomForestClassifier(n_estimators=400, max_depth=60)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
st.write("Random Forest's Accuracy with 400 estimators and max depth 60: %.3f" % accuracy)

# Save the trained scaler and model to reuse for prediction
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the scaler and model for prediction
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a form for user input
st.write("## Predict Bird Type")

with st.form(key='input_form'):
    st.write("### Input Features")
    feature_1 = st.number_input("Feature 1")
    feature_2 = st.number_input("Feature 2")
    feature_3 = st.number_input("Feature 3")
    feature_4 = st.number_input("Feature 4")
    feature_5 = st.number_input("Feature 5")
    feature_6 = st.number_input("Feature 6")
    feature_7 = st.number_input("Feature 7")
    feature_8 = st.number_input("Feature 8")
    feature_9 = st.number_input("Feature 9")
    feature_10 = st.number_input("Feature 10")
    feature_11 = st.number_input("Feature 11")
    feature_12 = st.number_input("Feature 12")
    feature_13 = st.number_input("Feature 13")
    feature_14 = st.number_input("Feature 14")
    feature_15 = st.number_input("Feature 15")
    feature_16 = st.number_input("Feature 16")

    submit_button = st.form_submit_button(label='Predict')

# Make prediction based on user input
if submit_button:
    user_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8,
                           feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16]])
    user_data = loaded_scaler.transform(user_data)
    prediction = loaded_model.predict(user_data)
    bird_types = ['astfly', 'bulori', 'warvir', 'woothr']
    predicted_bird = bird_types[prediction[0]]
    st.write(f"Predicted Bird Type: {predicted_bird}")
