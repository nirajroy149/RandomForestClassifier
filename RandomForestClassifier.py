import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa

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
x = StandardScaler().fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))
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

# Re-train the model with different parameters
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
st.write("Random Forest's Accuracy with 200 estimators: %.3f" % accuracy)

model = RandomForestClassifier(criterion="gini", max_depth=10)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
st.write("Random Forest's Accuracy with Gini criterion and max depth 10: %.3f" % accuracy)

# Predictions and evaluation
y_pred = model.predict(x_test)
st.write('Final Model Accuracy: %.3f' % accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.dataframe(cm)

# Heatmap for the confusion matrix
st.write("Confusion Matrix Heatmap:")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['astfly', 'bulori', 'warvir', 'woothr'], yticklabels=['astfly', 'bulori', 'warvir', 'woothr'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Truth')
st.pyplot(fig)
