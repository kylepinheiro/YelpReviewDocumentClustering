# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 23:59:35 2016

@author: pinhka12

Note: normalize the LDA topic vectors
"""

"""
FOR NEXT TIME:
-Learn about PCA for reducing dimensionality so you can visualize the k-means data
-look into other ways to cluster (matrix manipulations)
For future:
-Try using NMF to find topic models


"""
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
brs = pickle.load(open("coffee_business_review_corpora_without_french.p","rb"))




reviews = []
for key in brs.keys():
    reviews.append((key,brs[key]))
    
tfidf_vectorizer = TfidfVectorizer(max_df=0.6,analyzer = "word", stop_words = 'english',min_df=.05,max_features = 50000,sublinear_tf=True,) 
data_features = tfidf_vectorizer.fit([review[1] for review in reviews])
vocab = tfidf_vectorizer.get_feature_names()

#create a count_vectorizer to go over reviews again but only count the vocab
#that was found to be important by the tfidf vectorizer
count_vectorizer = CountVectorizer(analyzer = "word", vocabulary=vocab)

def run_kmeans(n,save=False,filename=""):
    tf_data_features = count_vectorizer.fit_transform([review[1] for review in reviews])
    #tf_data_features_array = tf_data_features.toarray()
    tf_vocab = count_vectorizer.get_feature_names() #to check that has same vocab
    
    
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_topics=17, max_iter=2,
                                    learning_method='online',learning_offset=10.,
                                    random_state=5)
    topic_transformed_features = lda.fit_transform(tf_data_features) #topic_transformed_features is array of topic composition of reviews
    
    #code taken from example on scikit learn to print
    for topic_idx, topic in enumerate(lda.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([tf_vocab[i] for i in topic.argsort()[:-50 - 1:-1]]))
        
    #normalize LDA topic score vectors
    topic_transformed_features = Normalizer(copy=False).fit_transform(topic_transformed_features)
    
    num_clusters = n
    k_means = cluster.KMeans(n_clusters=num_clusters)
    #run k means clustering algorithm
    k_means.fit(topic_transformed_features)
    
    original_space_centroids = k_means.cluster_centers_
    
    #print out the centroid topic scores for topics that scored above 0.10
    for i in range(num_clusters):
        print("Cluster %d: " % i, end='\n')
        for x in range(0,original_space_centroids[i].size):
            if original_space_centroids[i,x] > .1:
                print("Topic ",x,": ",original_space_centroids[i,x])
        print("")
        print("")
        
    #reduce dimensionality for visualization and silhouette score calculation
    svd = TruncatedSVD(2)
    data_features = svd.fit_transform(topic_transformed_features)
    space_centroids = svd.transform(k_means.cluster_centers_)
    
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
    if n >= 4:
        plt.plot([x[0] for x in fourthArray],[y[1] for y in fourthArray], 'mo',label="Cluster 3")
    
    
    plt.plot([centroid[0] for centroid in space_centroids], [centroid[1] for centroid in space_centroids],'ko')
    plt.title('K-Means Clustering with LDA Feature Vectorization ('+str(n)+' Clusters)')
    plt.legend(loc='upper right',shadow=True, fontsize='medium')    
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    if save is True and filename != "":
        plt.savefig(filename+'.png', dpi=100)
    plt.show()
    
    #calculate silhouette score
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
def plot_silhouette_scores(silhouette_scores_tuple): # try adding savefig('filename.png') to the end to save the figure
    cluster_numbers = [ t[0] for t in silhouette_scores_tuple]
    scores = [ t[1] for t in silhouette_scores_tuple]
    plt.plot(cluster_numbers, scores, 'ro')
    plt.title('K-Means Clustering with LDA Feature Vectorization:\n# Clusters vs Average Silhouette Score')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Silhouette Score")
    plt.gca().set_ylim([0,0.7])
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    #plt.savefig('LDA_k_means_silhouette_plot_without_french_v2.png',dpi=100)
    plt.show()
    
    