import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset

test = LoadDataset(which='bc')
X = test.data.data
feature_names = test.data.feature_names
target_names = test.data.target_names

# train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)
# np.save("X_train.npy", train)
# np.save("X_test.npy", test)
# np.save("y_train.npy", labels_train)
# np.save("y_test.npy", labels_test)

train = np.load("data/X_train.npy")
test = np.load("data/X_test.npy")
# print("test",test)
labels_train = np.load("data/y_train.npy")
labels_test = np.load("data/y_test.npy")

rf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5, max_features=5)
rf.fit(train, labels_train)
mean_accuracy = rf.score(test, labels_test)

explainer = LimeTabularExplainer(train,
                                 mode="classification",
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 discretize_continuous=True,
                                 verbose=False)

clustering = AgglomerativeClustering().fit(X)
names = list(feature_names)+["membership"]
clustered_data = np.column_stack([X, clustering.labels_])

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train)
distances, indices = nbrs.kneighbors(test)
clabel = clustering.labels_

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    print("inter",len(s1.intersection(s2)) / len(s1.union(s2))) # the higher this value is, the better (more intersection = more stability )
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_distance(usecase):
    sim = [] # list of lists that hold 10 elements per list 
    print(len(usecase))
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1-jaccard_similarity(l, j))
            # if you are trying to find the jaccard similarity, why are you finding it when l and j are the same? Then similarity will be 1 (always)
            # If you are doing 1- similarity, then you are finding the distance between the two lists. Lower distances = Better :)
        sim.append(i_sim)
        # print("isim", i_sim)
    # print(len(sim))
    return sim


for x in range(0, test.shape[0]): # for each value in test
    use_case_one_features = [] # Not used 
    use_case_two_features = []
    use_case_three_features = []
    use_case_four_features = [] # Not used
    for i in range(0, 10): # repeat 10 times 
        p_label = clabel[indices[x]] # predicted label of test[x]? 
        N = clustered_data[clustered_data[:, 30] == p_label] # get the records where the data == predicted label. 30 columns of features and 1 column of label
        subset = np.delete(N, 30, axis=1) # Delete the target column from subset 
        exp_dlime = explainer.explain_instance_hclust(test[x],
                                             rf.predict_proba,
                                             num_features=10,
                                             model_regressor=LinearRegression(),
                                             clustered_data = subset, # Note that the cluster gets put into here 
                                             regressor = 'linear', explainer='dlime', labels=(0,1))

        fig_dlime, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = i+.2, label='0') # it seems like we are dealing with only class 0 here? 
        # Also why are there so many of the same figures with different names? 

        # print("r_features", r_features)
        # fig_dlime.show()
        
        # We use 30 samples to define a cluster (which are the points that were given the same prediction as the label). We will put that into the DLIME explainer.
        # r_features is the result given back from dlime. This is the list containing the top 10 features that explain the prediction. I feel as if we don't need 
        # the for i in range(10). It might be better to just loop through the test set and seek the stability of each test example point. 
        use_case_two_features.append(r_features) 

        # Note that in LIME, there is no cluster data.
        exp_lime = explainer.explain_instance_hclust(test[x],
                                             rf.predict_proba,
                                             num_features=10,
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear', explainer = 'lime', labels=(0,1))

        fig_lime, r_features = exp_lime.as_pyplot_to_figure(type='h', name = i+.3, label='0')
        # fig_lime.show()
        use_case_three_features.append(r_features)
    # print(use_case_two_features)


    ################################################
    sim = jaccard_distance(use_case_two_features)
    # np.savetxt("results/rf_dlime_jdist_bc.csv", sim, delimiter=",")
    print("UC 2:",np.asarray(sim).mean())

    # plt.matshow(sim);
    # plt.colorbar()
    # plt.savefig("results/sim_use_case_2.pdf", bbox_inches='tight')
    # plt.show()

    ################################################
    sim = jaccard_distance(use_case_three_features)
    # np.savetxt("results/rf_lime_jdist_bc.csv", sim, delimiter=",")
    print("LIME",np.asarray(sim).mean())

    # plt.matshow(sim);
    # plt.colorbar()
    # plt.savefig("results/sim_use_case_3.pdf", bbox_inches='tight')
    # plt.show()
