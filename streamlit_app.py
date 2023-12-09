

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt

with st.sidebar:
    st.title('🤖 Machine Learning App v2')
    st.header('1. Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

    # Parameter settings
    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.select_slider('Max features (max_features)', options=[None, 'sqrt', 'log2'])
    parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
    parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


with st.status("Building model ...", expanded=True) as status:
    st.write("Loading data ...")
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    
    st.write("Preparing data ...")
    X = df.drop('logS', axis=1)
    y = df['logS']
    
    st.write("Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

    st.write("Training the model ...")
    rf = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        max_features=parameter_max_features,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        random_state=parameter_random_state,
        criterion=parameter_criterion,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
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
    parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
    if 'Mse' in parameter_criterion_string:
        parameter_criterion_string = parameter_criterion_string.replace('Mse', 'MSE')
    rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
    rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
    
    status.update(label="Model built!", state="complete", expanded=False)

col = st.columns(5)
col[0].metric(label="No. of samples", value=X.shape[0], delta="")
col[1].metric(label="No. of X variables", value=X.shape[1], delta="")

st.dataframe(rf_results)


