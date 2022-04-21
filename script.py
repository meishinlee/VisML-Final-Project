import streamlit as st
# streamlit run c:/Users/miche/Documents/GitHub/VisML-Final-Project/script.py
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset
HEP_DATA = "https://raw.githubusercontent.com/meishinlee/VisML-Final-Project/master/data/hepatitis.csv"

# test = LoadDataset(which='hp')
# X = test.data.data

X = df = pd.read_csv(HEP_DATA)
# feature_names = test.data.feature_names
feature_names = ['Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'Sgot', 'AlbuMin', 'ProTime', 'Histology', 'Class']
# target_names = test.data.target_names
target_names = ['yes' 'no']

# train = np.load("X_train_hp.npy")
# train = np.load("https://github.com/meishinlee/VisML-Final-Project/blob/7cdabbbec433278781e2a92758da0f30584c056e/data/X_train_hp.npy")
train = np.load("X_train_hp.npy")
test = np.load("data/X_test_hp.npy")
labels_train = np.load("data/y_train_hp.npy")
labels_test = np.load("data/y_test_hp.npy")

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(train, labels_train)
i = np.random.randint(0, test.shape[0])

st.write(""" 
# VisML Final Project
Mahika Jain, Mei Shin Lee  
""")

df = pd.read_csv(HEP_DATA)
print(df.head())
st.line_chart(df)