import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Iris_edit.csv', delimiter=',', usecols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'], header=None, skiprows=1, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Null values
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# handling the missing value
data = data.select_dtypes(include=[np.number]).interpolate().dropna()

#Visualize data in CSV file
sns.FacetGrid(data, hue='Species', height=4).map(plt.scatter, 'SepalLengthCm', 'PetalLengthCm')
plt.show()
sns.FacetGrid(data, hue='Species', height=4).map(plt.scatter, 'SepalWidthCm', 'PetalWidthCm')
plt.show()
sns.FacetGrid(data, hue='Species', height=4).map(plt.scatter, 'SepalWidthCm', 'PetalLengthCm')
plt.show()

# find the top correlated values
numeric_features = data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['Species'].sort_values(ascending=False)[:4], '\n')

# Preprocessing the data using scaler
scaler = preprocessing.StandardScaler()
scaler.fit(data)
X_scaled_array = scaler.transform(data)
X_scaled = pd.DataFrame(X_scaled_array, columns = data.columns)

wcss = []

# Finding k using the elbow method
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(data, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# the best score is shown for nK = 2
