import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt

with st.status("Building model ...", expanded=True) as status:
    st.write("Loading data ...")
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    
    st.write("Preparing data ...")
    X = df.drop('logS', axis=1)
    y = df['logS']
    
    st.write("Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    st.write("Training the model ...")
    rf = RandomForestRegressor(max_depth=2, random_state=100)
    rf.fit(X_train, y_train)

    st.write("Applying model to make predictions ...")
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    st.write("Evaluating performance metrics ...")
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    st.write("Displaying performance metrics ...")
    rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
    rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    
    status.update(label="Model built!", state="complete", expanded=False)


st.dataframe(rf_results)
