import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#add title
st.title('Data Analysis App')
st.subheader('Simple data analysis with Streamlit')

# Dropdown to select dataset
dataset_options = ['tips', 'iris', 'titanic']
selected_dataset = st.selectbox('Select a dataset', dataset_options)

# Button to upload own dataset
uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])

# Load the selected or uploaded dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    df = sns.load_dataset(selected_dataset)
    st.info(f"Loaded '{selected_dataset}' dataset from seaborn.")

# display the first few rows of the dataset
st.write("Dataset preview:", df.head())

# display number of rows and columns
st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")    

# display the column names with data types
st.write("Column names and data types:")
st.write(df.dtypes)

# print null values if greater than 0
if df.isnull().sum().sum() > 0:
    st.write("Null values in the dataset:")
    st.write(df.isnull().sum().sort_values(ascending=False))
    
#display basic statistics
st.write("Basic statistics of the dataset:")
st.write(df.describe())

# create a pairplot for numerical columns
st.subheader("Pairplot of numerical columns")

hue_column = st.selectbox("select a column to be used as hue: ", df.columns)
st.pyplot(sns.pairplot(df.select_dtypes(include=[np.number]), hue=hue_column))

#create a heatmap for correlation using numerical columns only
st.subheader("Correlation Heatmap of Numerical Columns")
if not df.select_dtypes(include=[np.number]).empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
