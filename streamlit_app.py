import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# Data preparation
X = df.drop('logS', axis=1)
y = df['logS']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Applying the model to make a prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

with st.status("Building model...", expanded=True) as status:
    st.write("Searching for data...")
    time.sleep(2)
  
    st.write("Found URL.")
    time.sleep(1)
  
    st.write("Downloading data...")
    time.sleep(1)
  
    status.update(label="Model built!", state="complete", expanded=False)
