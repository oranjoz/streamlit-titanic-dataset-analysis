# app.py
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Titanic EDA", layout="wide")

# Title
st.title("🚢 Titanic Dataset Exploratory Data Analysis")

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Display the dataset
st.subheader("Raw Titanic Dataset")
st.dataframe(df)

# Basic stats
st.subheader("Basic Dataset Information")
st.write("Shape of dataset:", df.shape)
st.write("Column types:")
st.write(df.dtypes)
st.write("Missing values:")
st.write(df.isnull().sum())

# Sidebar filters
st.sidebar.header("Filter Data")
sex = st.sidebar.selectbox("Select Gender", ["all"] + df["sex"].dropna().unique().tolist())
class_ = st.sidebar.selectbox("Select Passenger Class", ["all"] + df["class"].dropna().unique().tolist())

# Apply filters
filtered_df = df.copy()
if sex != "all":
    filtered_df = filtered_df[filtered_df["sex"] == sex]
if class_ != "all":
    filtered_df = filtered_df[filtered_df["class"] == class_]

# Display filtered dataset
st.subheader("Filtered Dataset")
st.write(f"Showing {len(filtered_df)} rows after filtering")
st.dataframe(filtered_df)

# Survival count plot
st.subheader("Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="survived", hue="sex", ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Died", "Survived"])
st.pyplot(fig1)

# Age distribution
st.subheader("Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df["age"].dropna(), bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)
