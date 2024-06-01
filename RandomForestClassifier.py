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

# Feature names
feature_names = [
    "zcr_mean", "zcr_std", "spectral_centroid", "mean_spectral_rolloff", "std_spectral_rolloff",
    "spectral_bandwidth", "chroma1", "chroma2", "chroma3", "chroma4", "chroma5", "chroma6", 
    "chroma7", "chroma8", "chroma9", "chroma10", "chroma11", "chroma12", "mfcc1", "mfcc2", "mfcc3", 
    "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", 
    "mfcc14", "mfcc15", "mfcc16", "mfcc17", "mfcc18", "mfcc19", "mfcc20", "mfcc21", "mfcc22", 
    "mfcc23", "mfcc24", "mfcc25", "mfcc26"
]

# Default values for the features
default_values = [0.351090641, 0.051201837, 4376.084267, 6666.24838, 1496.420809, 2154.197311, 
                  0.280201048, 0.404124886, 0.513666272, 0.663726926, 0.345381081, 0.305490345, 
                  0.273532033, 0.281354696, 0.24505122, 0.312662661, 0.247952819, 0.245751858, 
                  -542.4055176, -94.81381226, -83.49885559, 32.07971191, 24.36644554, 17.66520309, 
                  -9.617711067, -21.92518806, 1.593942881, 2.228652, 10.17960167, 10.20607471, 
                  -16.90008926, -0.006378755, -3.456950188, 4.543880939, 9.178504944, 
                  -3.597657681, -6.942389965, 2.342945099, -4.922445774, 10.15924072, 
                  -1.919533253, -1.433038712, -3.299443483, 0.0665804]

# Create a form for user input
st.write("## Predict Bird Type")

with st.form(key='input_form'):
    st.write("### Input Features")
    feature_inputs = [st.number_input(feature_names[i], value=default_values[i]) for i in range(len(feature_names))]
    submit_button = st.form_submit_button(label='Predict')  # Moved submit button here


with st.form(key='input_form'):
    st.write("### Input Features")
    feature_inputs = [st.number_input(feature_names[i], value=default_values[i]) for i in range(len(feature_names))]
    submit_button = st.form_submit_button(label='Predict')  # Moved submit button here

    # Button to generate random values
    if st.button("Generate Random Values"):
        random_values = generate_random_values()
        for i in range(len(feature_names)):
            feature_inputs[i].number_input(label=feature_names[i], value=random_values[i])


# Make prediction based on user input
if submit_button:
    user_data = np.array([feature_inputs])
    user_data = loaded_scaler.transform(user_data)
    prediction = loaded_model.predict(user_data)
    bird_types = ['astfly', 'bulori', 'warvir', 'woothr']
    st.write("Predicted Bird Type:", bird_types[prediction[0]])