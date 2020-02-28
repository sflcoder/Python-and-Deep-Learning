import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')

x = dataset.iloc[:,1:17]
y = dataset.iloc[:,-1]
print(x)

dataset = dataset.fillna(dataset.mean())

x = dataset.iloc[:,1:17]
y = dataset.iloc[:,-1]
print(x)

print (x.shape)
print (y.shape)

wcss = []  ##Within Cluster Sum of Squares##elbow method to know the number of clusters
for i in range(1,16):
    kmeans = KMeans(n_clusters=i,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,16),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

##building the model
from sklearn.cluster import KMeans
nclusters = 6 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('silhouette score: ',score)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)
x_scaler = scaler.transform(x)
km.fit(x_scaler)
y_cluster_kmeans = km.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
print('silhouette score(With feature scaling): ',score)


pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2,dataset[['TENURE']]],axis=1)
#print(finaldf)

km.fit(finaldf)
y_cluster_kmeans = km.predict(finaldf)
score = metrics.silhouette_score(finaldf, y_cluster_kmeans)
print('silhouette score(PCA): ',score)

