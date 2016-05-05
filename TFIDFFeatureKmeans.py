# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 23:59:35 2016

@author: pinhka12


"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster, metrics
import matplotlib.pyplot as plt

#load reviews from pickle file
#this is a file containing a dictionary with business id as the key and the value 
#being a string that is the concatenation of all reviews for that business
brs = pickle.load(open("coffee_business_review_corpora_without_french.p","rb"))

#create an array of just the review texts
reviews = []
for key in brs.keys():
    reviews.append((key,brs[key]))

#Create feature vectors based on TF-IDF scores
tfidf_vectorizer = TfidfVectorizer(max_df=0.6,analyzer = "word",stop_words = 'english', min_df=0.05,sublinear_tf=True,max_features = 50000,norm="l2")
"""
PARAMETERS:
n (number of clusters)
save (write image of graph to a file or not)
filename (file to write graph to if save is true)

OUTPUT:
silhouette score of the clustering
"""
  
def run_kmeans(n,save=False,filename=""):
    data_features = tfidf_vectorizer.fit_transform([review[1] for review in reviews])
    
    data_features_array = data_features.toarray()
    
    vocab = tfidf_vectorizer.get_feature_names()
    
    num_clusters = n 
    k_means = cluster.KMeans(n_clusters = num_clusters,init='random',
                             max_iter=100,n_init=10)
            
    k_means.fit(data_features) #run k_means clustering algorithm
    
    original_space_centroids = k_means.cluster_centers_ # hang on to centroid
                                                        #feature vectors
    order_centroids = original_space_centroids.argsort()[:,::-1] 
    #for each centroid, order its terms by TF-IDF score
    
    #prints out 50 highest TF-IDF scoring terms from each centroid
    for i in range(num_clusters):
        print("Cluster %d: " % i, end=' ')
        for ind in order_centroids[i,:50]:
            print('%s' % vocab[ind],end=' ')
        print("")
        print("")
        
    #dimension reduction
    svd = TruncatedSVD(2)
    data_features = svd.fit_transform(data_features)
    space_centroids = svd.transform(k_means.cluster_centers_)
    
    #for producing colored document cluster graphs
    data_features_array = data_features.tolist()
    firstArray = []
    secondArray = []
    thirdArray = []
    fourthArray = []
    for i in range(len(data_features_array)):
        #print(coord_pair)
        if k_means.labels_[i] == 0:
            firstArray.append((data_features_array[i][0],data_features_array[i][1]))
        elif k_means.labels_[i] == 1:
            secondArray.append((data_features_array[i][0],data_features_array[i][1]))
        elif k_means.labels_[i] == 2:
            thirdArray.append((data_features_array[i][0],data_features_array[i][1]))
        else:
            fourthArray.append((data_features_array[i][0],data_features_array[i][1]))
    plt.plot([x[0] for x in firstArray],[y[1] for y in firstArray], 'ro',label="Cluster 0")
    if n >= 2:
        plt.plot([x[0] for x in secondArray],[y[1] for y in secondArray], 'go',label="Cluster 1")
    if n >= 3:
        plt.plot([x[0] for x in thirdArray],[y[1] for y in thirdArray], 'bo',label="Cluster 2")
    if n == 4:
        plt.plot([x[0] for x in fourthArray],[y[1] for y in fourthArray], 'mo',label="Cluster 3")
    
    
    plt.plot([centroid[0] for centroid in space_centroids], [centroid[1] for centroid in space_centroids],'ko')
    plt.title('K-Means Clustering with TF-IDF Feature Vectorization ('+str(n)+' Clusters) \n')
    plt.legend(loc='upper right',shadow=True, fontsize='medium')    
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    if save is True and filename != "": #if specified in parameters, save the graph as a png with 
                     #the provided filename
        plt.savefig(filename+".png", dpi=100)
    
    plt.show()
    #calculate silhouette score from dimensionally reduced dataset on  euclidean metric
    silhouette_score = metrics.silhouette_score(data_features,k_means.labels_,metric='euclidean',sample_size=len(reviews))
    print(num_clusters, ': silhouette score: ',silhouette_score)
    return silhouette_score
    
"""
Parameters:
lowerBound: starting number of clusters to try
upperBound: exclusive upper bound of number of clusters to run k_means with

Output:
array of tuples of the form (#clusters,silhouette score)
"""
def get_silhouette_scores(lowerBound,upperBound, numRepetitions):
    silhouette_scores = []
    for x in range(lowerBound,upperBound):
        total = 0
        for y in range(0,numRepetitions):
            total+=run_kmeans(x)
        average = total/numRepetitions
        silhouette_scores.append((x, average))
    return silhouette_scores
    
"""
Parameters:
silhouette_scores_tuple: array with tuples of form (# clusters, silhouette score for that # clusters)
                        likely the output from 'get_silhouette_scores'
save: specify True if plot should be saved
filename: if save is True, specify the filename to be saved to (without extension)

"""
def plot_silhouette_scores(silhouette_scores_tuple,save=False,filename=""):
    cluster_numbers = [ t[0] for t in silhouette_scores_tuple]
    scores = [ t[1] for t in silhouette_scores_tuple]
    plt.plot(cluster_numbers, scores, 'ro')
    plt.title('K-Means Clustering with TF-IDF Feature Vectorization:\n# Clusters vs Average Silhouette Score')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Silhouette Score")
    plt.gca().set_ylim([0,0.7])
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    if save is True:
        plt.savefig(filename+'.png',dpi=100)
    plt.show()
    