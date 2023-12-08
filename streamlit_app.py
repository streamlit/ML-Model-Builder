import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt

with st.status("Building model...", expanded=True) as status:
    st.write("Loading data...")
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    
    st.write("Preparing data")
    X = df.drop('logS', axis=1)
    y = df['logS']
    
    st.write("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    st.write("Training the model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    st.write("Applying model to make predictions...")
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    
    # Evaluating performance metrics
    st.write("Evaluating performance metrics")
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)


    
    status.update(label="Model built!", state="complete", expanded=False)

