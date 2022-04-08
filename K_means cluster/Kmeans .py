import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:2000,[0, 1]].values
X.shape



# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
ilist = []
n = 15
for i in range(1,n):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    ilist.append(kmeans.inertia_)
    
plt.plot(range(1,n), ilist)
plt.title('Elbow')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertias')
plt.show()



# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)



#the position of cluster centers
print('KMeansModel centers are :\n ' , kmeans.cluster_centers_)




# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'r')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'b')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'g')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'c')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'm')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'y')
plt.title('Clusters of Houses')
plt.xlabel('Area')
plt.ylabel('Value')
plt.legend()
plt.show()
