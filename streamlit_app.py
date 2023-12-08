import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


with st.status("Building model...", expanded=True) as status:
    # Loading data
    st.write("Loading data...")
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    
    # Preparing data
    st.write("Preparing data")
    X = df.drop('logS', axis=1)
    y = df['logS']
    
    # Splitting data
    st.write("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    
    status.update(label="Model built!", state="complete", expanded=False)

