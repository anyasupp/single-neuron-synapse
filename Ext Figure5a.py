# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:27:28 2022
Cluster neurons into subtypes according to their morphology
See Nikolaou, N. & Meyer, M. P. Lamination Speeds the Functional Development of Visual Circuits. Neuron 88, 999–1013 (2015).
@author: AnyaS
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

dt = pd.read_excel("Ext Figure5.xlsx",sheet_name='raw')  
#%%prep data
print(dt.isna().sum())
#fill Nan with 0 
dt['PA_loc'].fillna(0, inplace = True)

X = np.array(dt.drop(['Fish_ID','Condition','Distance_Skin','Darbour_distance','PA_distance',
                        'PA_distance','Total','Darbour_Thickness_'], 1).astype(float))
# df = dt.drop(['Fish_ID','Condition','Distance_Skin','Darbour_distance','PA_distance',
#                         'PA_distance','Total','Darbour_Thickness_'], 1).astype(float)

#%%REDUCE DIMENSIONALITY
#try to find optimal number of components which capture the greatest amount of variance in the data
# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(X)
# Create a PCA instance
pca = PCA()
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

#cumulative variance plot
pca.explained_variance_ratio_
plt.figure(figsize= (10,8))
plt.plot(range(0,6),pca.explained_variance_ratio_.cumsum(), marker ='o',linestyle='--')
x_label = [1,2,3,4,5,6]
plt.xticks(range(0,6),x_label)
plt.title('Explained Variance by Components')
plt.xlabel('No of components')
plt.ylabel('Cumulative Explained Variance')
#
#%%
#perform PCA with the chosen number of components
# choosing 4 as over 80% of variance
pca= PCA(n_components= 4)
pca.fit(X_std)
#calculate resulting components scores for the elements in our dataset 
pca.transform(X_std)
scores_pca_4= pca.transform(X_std)
#incorporate newly obtained PCA scores in the K-means algorithm 

#do the elbow method 
wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init ='k-means++', random_state=22)
    kmeans_pca.fit(scores_pca_4)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,11),wcss,marker='o',linestyle='-')
plt.tick_params(axis='both', which='major', labelsize=14)  
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow: K-means with PCA Clustering (4components)')
plt.show()

#%%Yelbow method using yellow brick
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12),locate_elbow=True,timings=False)
#green line shows time to train clustering model per K (to hide it timings=False)
visualizer.fit(scores_pca_4)
visualizer.poof()

#%%use calinski harabasz
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,12), metric='calinski_harabasz', timings=False, locate_elbow=False
)
visualizer.fit(scores_pca_4)        # Fit the data to the visualizer
visualizer.show()     

#%%
#use silhouette
#Silhouette analysis use to evaluate density and separation between clusters - average silhouette coefficient for each sample
#difference between intra-cluster distance and the mean nearest-cluster distance -- then normalized
#scores near +1 = high separation 
#scores near -1 indicate samples may be assigned to wrong cluster
#put in number of clusters and see distribution
fig, ax = plt.subplots(1,4)
plt.subplot(1,4,1)
model = KMeans(3)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,2)
model = KMeans(4)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,3)
model = KMeans(5)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,4)
model = KMeans(6)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()
#plt.savefig('20200507_silhouette_2_for_withoutoutliers.jpg', format='jpg', dpi=1200)
#%%# Looks like PCA4 and k means 4 gives the best! 
# Proceed to do clustering
#use initializer and random state as before 
kmeans_pca = KMeans(n_clusters =4, init ='k-means++',random_state = 22)
#we fir our data with k-means pca model
kmeans_pca.fit(scores_pca_4)

#how to analyze PCA and K-means results

#create a new dataframe with the orginical features and add the pCA scores and assigned clusters
dt_segm_pca_kmeans = pd.concat([dt.reset_index(drop=True),pd.DataFrame(scores_pca_4)], axis=1)
dt_segm_pca_kmeans.columns.values[-4:] = ['Component1','Component2','Component3','Component4']
#last column add pca k-means cluster labels
dt_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

#see head of our new dataframe
dt_segm_pca_kmeans.head()

#add names of segments to the labels
dt_segm_pca_kmeans['Segment'] = dt_segm_pca_kmeans['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth'})
