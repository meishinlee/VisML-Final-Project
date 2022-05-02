import streamlit as st
# streamlit run c:/Users/miche/Documents/GitHub/VisML-Final-Project/script.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import io 

# from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset
HEP_DATA = "https://raw.githubusercontent.com/meishinlee/VisML-Final-Project/master/data/hepatitis.csv"
BC_DATA = "https://github.com/meishinlee/VisML-Final-Project/blob/master/X_train_hp.npy"

# def load_data(): 
#     return np.random.rand(455, 30)

# train = load_data()
# Create an in-memory buffer
# with io.BytesIO() as buffer:
#     np.save(buffer, train)
#     btn = st.download_button(
#         label="Download numpy array (.npy)",
#         data = buffer, # Download buffer
#         file_name = 'predicted_map.npy'
#     )
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
labels_train = np.load("data/y_train.npy")

labels_train = np.load("data/y_train_hp.npy")
labels_test = np.load("data/y_test_hp.npy")

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(train, labels_train)
i = np.random.randint(0, test.shape[0])

st.write(""" 
# VisML Final Project
Mahika Jain, Mei Shin Lee  
""")

st.write('''
Breast Cancer Dataset 
''')
fig, ax = plt.subplots()
ax = plt.scatter(train[:,0], train[:,17], c=labels_train)# labels_train)
st.pyplot(fig)

# In this case, it was okay to cluster the entire dataset for visualization purposes. We were not training 
# and testing a model. 
test = LoadDataset(which='bc')
X = test.data.data
feature_names = test.data.feature_names
target_names = test.data.target_names

st.write("Agglomerative Clustering")
# Agglomerative Clustering
clustering = AgglomerativeClustering().fit(X)
clustered_data = np.column_stack([X, clustering.labels_])
clabel = clustering.labels_
fig_agglomerative, ax = plt.subplots()
feature1 = st.slider("Feature 1: ", 1, 17, 1, key="agg_1")
feature2 = st.slider("Feature 2: ", 1, 17, 2, key="agg_2")
ax = plt.scatter(X[:,feature1], X[:,feature2], c=clabel, cmap='rainbow')# labels_train)
st.pyplot(fig_agglomerative)

# EM Clustering 
clustering = GaussianMixture(n_components=2, init_params='kmeans').fit(X)
# names = list(feature_names)+["membership"]
clustered_data = np.column_stack([X, clustering.predict(X)])
fig_em, ax = plt.subplots()
# print(clustering.predict(X))
st.write("EM Clustering")
# We start with index 1 because index 0 is the key/index of the samples
feature1 = st.slider("Feature 1: ", 1, 29, 1, key="em_1")
feature2 = st.slider("Feature 2: ", 1, 29, 2, key="em_2")
ax = plt.scatter(X[:,feature1], X[:,feature2], c=clustering.predict(X), cmap='rainbow')# labels_train)
plt.xlabel(feature_names[feature1])
plt.ylabel(feature_names[feature2])
st.pyplot(fig_em)

# Testing streamlit 
train = pd.DataFrame(X)
train.columns=feature_names
train.reset_index(inplace=True)
st.write(train)

df = pd.read_csv(HEP_DATA)
print(df.head())
st.line_chart(df)