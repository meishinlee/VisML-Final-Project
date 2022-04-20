import streamlit as st
import pandas as pd

st.write(""" 
# VisML Final Project
Mei Shin Lee, Mahika Jain  
""")

HEP_DATA = "https://raw.githubusercontent.com/meishinlee/VisML-Final-Project/master/data/hepatitis.csv"
df = pd.read_csv(HEP_DATA)
print(df.head())
st.line_chart(df)