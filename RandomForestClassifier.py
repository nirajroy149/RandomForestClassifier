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

# Feature names and default values
feature_names = [
    "zcr_mean", "zcr_std", "spectral_centroid", "mean_spectral_rolloff", "std_spectral_rolloff",
    "spectral_bandwidth", "chroma1", "chroma2", "chroma3", "chroma4", "chroma5", "chroma6", 
    "chroma7", "chroma8", "chroma9", "chroma10", "chroma11", "chroma12", "rms", "spectral_contrast1", 
    "spectral_contrast2", "spectral_contrast3", "spectral_contrast4", "spectral_contrast5", 
    "spectral_contrast6", "spectral_contrast7", "tonnetz1", "tonnetz2", "tonnetz3", "tonnetz4", 
    "tonnetz5", "tonnetz6", "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", 
    "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", "mfcc14", "mfcc15", "mfcc16", 
    "mfcc17", "mfcc18", "mfcc19", "mfcc20", "mfcc21", "mfcc22", "mfcc23", "mfcc24", "mfcc25", "mfcc26"
]

default_values = [
    0.351091, 0.051202, 4376.084267, 6666.248380, 1496.420809, 2154.197311, 0.280201, 0.404125, 
    0.513666, 0.619383, 0.497849, 0.385019, 0.298793, 0.333074, 0.407158, 0.412058, 0.398303, 
    0.431288, 0.205087, 21.940492, 21.854172, 22.125953, 22.079578, 22.085982, 22.028218, 
    22.095961, -1.352716, -1.606311, -1.381885, -1.587543, -1.583343, -1.406742, -3.675927, 
    4.868219, 2.869121, -4.413172, -2.730347, -1.612245, 1.525317, -3.346588, 1.095379, 
    -3.545896, -2.641084, -3.205458, 0.035301, -3.169244, -3.262197, -3.597658, -6.942390, 
    2.342945, -4.922446, 10.159241, -1.919533, -1.433039, -3.299443, 0.066580 
]

with st.form(key='input_form'):
    st.write("### Input Features")
    feature_inputs = [st.number_input(feature_names[i], value=default_values[i]) for i in range(len(feature_names))]
    submit_button = st.form_submit_button(label='Predict')  # Moved submit button here

# Make prediction based on user input
if submit_button:
    user_data = np.array([feature_inputs])
    user_data = loaded_scaler.transform(user_data)
    prediction = loaded_model.predict(user_data)
    bird_types = ['astfly', 'bulori', 'warvir', 'woothr']
    predicted_bird = bird_types[prediction[0]]
    st.write(f"Predicted Bird Type: {predicted_bird}")
