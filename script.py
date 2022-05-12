from re import M
import streamlit as st
import streamlit.components.v1 as components
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
import lime
import lime.lime_tabular
from sklearn.neural_network import MLPClassifier
# from explainer_tabular import LimeTabularExplainer

# from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset
HEP_DATA = "https://raw.githubusercontent.com/meishinlee/VisML-Final-Project/master/data/hepatitis.csv"
BC_DATA = "https://github.com/meishinlee/VisML-Final-Project/blob/master/X_train_hp.npy"

X = df = pd.read_csv(HEP_DATA)
# feature_names = test.data.feature_names
feature_names = ['Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'Sgot', 'AlbuMin', 'ProTime', 'Histology', 'Class']
# target_names = test.data.target_names
target_names = ['yes' 'no']

# train = np.load("X_train_hp.npy")
# train = np.load("https://github.com/meishinlee/VisML-Final-Project/blob/7cdabbbec433278781e2a92758da0f30584c056e/data/X_train_hp.npy")
# train = np.load("X_train_hp.npy")
# test = np.load("data/X_test_hp.npy")
# labels_train = np.load("data/y_train.npy")

# labels_train = np.load("data/y_train_hp.npy")
# labels_test = np.load("data/y_test_hp.npy")

# rf = RandomForestClassifier(n_estimators=10, random_state=0)
# rf.fit(train, labels_train)
# i = np.random.randint(0, test.shape[0])

st.write(""" 
# Modified DLIME Experimental Results
## CS-UY 3943, Final Project
### Mahika Jain, Mei Shin Lee  
""")

st.write('''
## Breast Cancer Dataset 
#### Clusters between the two classes (Malignant and Benign) are shown using varying features below. 
''')
# fig, ax = plt.subplots()
# ax = plt.scatter(train[:,0], train[:,17], c=labels_train)# labels_train)
# st.pyplot(fig)

# In this case, it was okay to cluster the entire dataset for visualization purposes. We were not training 
# and testing a model. 

test = LoadDataset(which='bc')
X = test.data.data
feature_names = test.data.feature_names
target_names = test.data.target_names

train = np.load("data/X_train.npy")
test = np.load("data/X_test.npy")
# print(test.shape)
labels_train = np.load("data/y_train.npy")
labels_test = np.load("data/y_test.npy")

st.write("# Agglomerative Clustering")
# Agglomerative Clustering
clustering = AgglomerativeClustering().fit(train)
clustered_data = np.column_stack([train, clustering.labels_])
clabel = clustering.labels_
fig_agglomerative, ax = plt.subplots()
feature1 = st.slider("Feature 1 (x-axis): ", 1, 17, 1, key="agg_1")
feature2 = st.slider("Feature 2 (y-axis): ", 1, 17, 2, key="agg_2")
plt.xlabel(feature_names[feature1])
plt.ylabel(feature_names[feature2])
ax = plt.scatter(train[:,feature1], train[:,feature2], c=clabel)# labels_train)
st.pyplot(fig_agglomerative)

# EM Clustering 
clustering = GaussianMixture(n_components=2, init_params='kmeans').fit(train)
# names = list(feature_names)+["membership"]
clustered_data = np.column_stack([train, clustering.predict(train)])
fig_em, ax = plt.subplots()
# print(clustering.predict(X))

st.write("# EM Clustering")
# We start with index 1 because index 0 is the key/index of the samples
# feature1 = st.slider("Feature 1: ", 1, 17, 1, key="em_1")
# feature2 = st.slider("Feature 2: ", 1, 17, 2, key="em_2")
ax = plt.scatter(train[:,feature1], train[:,feature2], c=clustering.predict(train))# labels_train)
plt.xlabel(feature_names[feature1])
plt.ylabel(feature_names[feature2])
st.pyplot(fig_em)

# For EM Clustering, Choose a random sample from the test set 
clabel = clustering.predict(X)
point_index = 1
point = test[point_index]
point_pred_label = clustering.predict(point.reshape(1, -1)) # this is the predicted label of the point
# ############################ HERE ##############################
# filter out the points that are in the same cluster as the point. Then run KNN to find the 10 closest out of 
# those points. Currently, KNN also picks up on points from opposing clusters. 

# Determine its class based on KNN
# Predict the class of the sample using the trained model

pred_same_index_list = [] # store the list of indices of the points that are in the same cluster as the test point (1 or 0 value)
for i in range(len(labels_train)): 
    if labels_train[i] == point_pred_label:
        pred_same_index_list.append(i)

# point_clustered_data contains the data of the points that are in the same cluster as the test point
point_clustered_data = train[pred_same_index_list]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train)
distances, indices = nbrs.kneighbors(test)
closest_point_dist = distances[point_index]
closest_point_index = indices[point_index]
closest_point = train[closest_point_index][-1] 
print("predicted label: ", point_pred_label)

n_neighbor = 10
ten_neighbors = NearestNeighbors(n_neighbors=n_neighbor, algorithm='ball_tree').fit(train)
distances_ten, indices_ten = ten_neighbors.kneighbors(test)
ten_closest_point_dist = distances_ten[point_index]
ten_closest_point_index = indices_ten[point_index]

fig_ten_closest, ax = plt.subplots()
ten_closest_point_data = train[ten_closest_point_index, :]  
ax = plt.scatter(ten_closest_point_data[:,feature1], ten_closest_point_data[:,feature2], c=clustering.predict(ten_closest_point_data), cmap='rainbow')# labels_train)
plt.xlabel(feature_names[feature1])
plt.ylabel(feature_names[feature2])

print(ten_closest_point_data[feature1])

# Plot Linear Regression model on selected points/features. This is what dLIME does with EM Clustering... but on 1 feature
min_x = min(ten_closest_point_data[:,feature1]) - 5
max_x = max(ten_closest_point_data[:,feature1]) + 5

# new_features = np.hstack((np.ones((len(ten_closest_point_data[feature1]), 1)), (ten_closest_point_data[feature1].reshape(len(ten_closest_point_data[feature1]), 1))))
reg = LinearRegression().fit(ten_closest_point_data[:,feature1].reshape(len(ten_closest_point_data[:,feature1]), 1), ten_closest_point_data[:,feature2].reshape(len(ten_closest_point_data[:,feature1]), 1))
coeff = reg.coef_
intercept = reg.intercept_
x_values = np.linspace(min_x, max_x, 100).reshape(100, 1)
ax = plt.plot(x_values, reg.predict(x_values))
st.pyplot(fig_ten_closest)

# Testing regular LIME 
st.write("# LIME results, with top 5 Features")
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(train, labels_train)

mean_accuracy = nn.score(test, labels_test)

from explainer_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(train,
                                 mode="classification",
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 discretize_continuous=True,
                                 verbose=False)

# explainer = lime.lime_tabular.LimeTabularExplainer(train,
#                                  mode="classification",
#                                  feature_names=feature_names,
#                                  class_names=target_names,
#                                  discretize_continuous=True,
#                                  verbose=False)
st.write("Trial 1")                                 
exp_lime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear', explainer = 'lime', labels=(0,1))

# print("pt",point.reshape((30,1)))
fig_lime, r_features = exp_lime.as_pyplot_to_figure(type='h', name = 1+.3, label='0')
st.pyplot(fig_lime)

st.write("Trial 2")
exp_lime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear', explainer = 'lime', labels=(0,1))

# print("pt",point.reshape((30,1)))
fig_lime2, r_features = exp_lime.as_pyplot_to_figure(type='h', name = 2+.3, label='0')
st.pyplot(fig_lime2)

st.write("Trial 3")
exp_lime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear', explainer = 'lime', labels=(0,1))

# print("pt",point.reshape((30,1)))
fig_lime3, r_features = exp_lime.as_pyplot_to_figure(type='h', name = 2+.3, label='0')
st.pyplot(fig_lime3)

# exp = exp_lime.explain_instance(point, nn.predict_proba, num_features=5)

# components.html(exp.as_html())
# fig_exp, ax = plt.subplots()
# exp_fig = exp.as_pyplot_figure()
# st.write(exp.as_list())

# If we run the LIME explainer as coded above with the same point (but different random state), 
# we can show that we would get different results... which proves the instability of the LIME 
# explainer.

# Now, we can also show this for DLIME... and hopefully it shows similar results based on clustering... 

st.write("# DLIME results (using EM Clustering), with top 5 Features")

st.write("Trial 1")
exp_dlime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor=LinearRegression(),
                                             clustered_data = point_clustered_data,
                                             regressor = 'linear', explainer='dlime', labels=(0,1))
fig_dlime1, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = 1+.2, label='0')
st.pyplot(fig_dlime1)

st.write("Trial 2")
exp_dlime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor=LinearRegression(),
                                             clustered_data = point_clustered_data,
                                             regressor = 'linear', explainer='dlime', labels=(0,1))
fig_dlime2, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = 2+.2, label='0')
st.pyplot(fig_dlime2)

st.write("Trial 3")
exp_dlime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor=LinearRegression(),
                                             clustered_data = point_clustered_data,
                                             regressor = 'linear', explainer='dlime', labels=(0,1))
fig_dlime3, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = 3+.2, label='0')
st.pyplot(fig_dlime3)

trial_num = st.number_input('Enter the trial number  you would like to see:', min_value=1, max_value=100000, step=1)
st.write('Trial ', trial_num)
exp_dlime = explainer.explain_instance_hclust(point,
                                             nn.predict_proba,
                                             num_features=5,
                                             model_regressor=LinearRegression(),
                                             clustered_data = point_clustered_data,
                                             regressor = 'linear', explainer='dlime', labels=(0,1))
fig_dlimenum, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = trial_num+.2, label='0')
st.pyplot(fig_dlimenum)

# Testing streamlit 
train = pd.DataFrame(X)
train.columns=feature_names
train.reset_index(inplace=True)
st.write(train)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
st.write("# Census Dataset")
CENSUS_DATA = "https://raw.githubusercontent.com/meishinlee/VisML-Final-Project/master/data/census-income-data.csv"
census_df = pd.read_csv(CENSUS_DATA)
census_df.drop("state of previous residence", axis=1, inplace=True)
st.write(census_df.head())

le = LabelEncoder()
categories_data = ['class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'tax filer stat', 'region of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'live in this house 1 year ago', 'family members under 18', 'citizenship', "fill inc questionnaire for veteran\'s admin"]
for i in range(len(categories_data)):
    #st.write(le.fit_transform(census_df[categories_data[i]])
    census_df[categories_data[i]] = le.fit_transform(census_df[categories_data[i]])

# census_df["class of worker"] = le.fit_transform(census_df["class of worker"])

# to_encode_cat = census_df[['class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'tax filer stat', 'region of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'live in this house 1 year ago', 'family members under 18', 'citizenship', "fill inc questionnaire for veteran\'s admin"]]
# to_encode_cat['detailed industry recode'] = to_encode_cat['detailed industry recode'].astype(str)
# to_encode_cat['detailed occupation recode'] = to_encode_cat['detailed occupation recode'].astype(str)

# selected_cat = [['class of worker'], ['detailed industry recode'], ['detailed occupation recode'], ['education'], ['enroll in edu inst last wk'], ['marital stat'], ['major industry code'], ['major occupation code'], ['race'], ['hispanic origin'], ['sex'], ['member of a labor union'], ['reason for unemployment'], ['full or part time employment stat'], ['tax filer stat'], ['region of previous residence'], ['detailed household and family stat'], ['detailed household summary in household'], ['live in this house 1 year ago'], ['family members under 18'], ['citizenship'], ["fill inc questionnaire for veteran\'s admin"]]
# st.write(to_encode_cat)
#enc = OneHotEncoder(categories = selected_cat, sparse=True, handle_unknown='ignore')
#encoded_census = enc.fit_transform(to_encode_cat).toarray()
# enc_fit = enc.fit(to_encode_cat)
# encoded_census = enc.transform(enc_fit).toarray()
# st.write(encoded_census)
st.write(census_df)

selected_nums = ['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks', 'num persons worked for employer', 'year', 'weeks worked in year', 'instance weight']
to_join = census_df[selected_nums]

#total = np.hstack((encoded_census, to_join))
total = np.hstack((census_df, to_join))
st.write(total)

# st.write(enc.categories_)

census_train, census_test, census_train_label, census_test_label = train_test_split(total, census_df['class of worker'], test_size=0.2, random_state=42)
st.write("# Agglomerative Clustering")
# Agglomerative Clustering
clustering = AgglomerativeClustering().fit(census_train)
clustered_data = np.column_stack([census_train, clustering.labels_])
clabel = clustering.labels_
fig_agglomerative, ax = plt.subplots()
feature1 = st.slider("Feature 1 (x-axis): ", 1, 17, 1, key="census_agg_1")
feature2 = st.slider("Feature 2 (y-axis): ", 1, 17, 2, key="census_agg_2")
plt.xlabel(feature_names[feature1])
plt.ylabel(feature_names[feature2])
ax = plt.scatter(train[:,feature1], train[:,feature2], c=clabel)# labels_train)
st.pyplot(fig_agglomerative)

df = pd.read_csv(HEP_DATA)
st.line_chart(df)