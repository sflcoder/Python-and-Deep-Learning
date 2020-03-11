import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

# Load data fromwinequalityN.csv
wineQuality = pd.read_csv('winequalityN.csv')
print('The data shape of wineQuality DataFrame:', wineQuality.shape)
print(wineQuality.describe())

# Handling null values
nullValues = pd.DataFrame(wineQuality.isnull().sum().sort_values(ascending=False))
nullValues.columns = ['Null Number']
nullValues.index.name = 'Feature'
print(nullValues)
print(wineQuality.shape)

# Replace the null values in the wineQuality by the mean
wineQuality = wineQuality.fillna(wineQuality.mean())
nullValues = pd.DataFrame(wineQuality.isnull().sum().sort_values(ascending=False))
nullValues.columns = ['Null Number']
nullValues.index.name = 'Feature'
print('\nAfter replacing the null values in the wineQuality DataFrame by the mean')
print(nullValues)

print(wineQuality.columns.values)
# see how many samples we have of each species
print(wineQuality["quality"].value_counts())
print(wineQuality["type"].value_counts())

numeric_features = wineQuality.select_dtypes(include=[np.number])
print('\nThe numeric features: ')
print(numeric_features.columns.values)

# Encoding a categorical feature
categorical_feature = wineQuality.select_dtypes(include='object')
print('\nThe non-numeric features ')
print(categorical_feature.columns.values)
# ['type'] is a categorical feature
print(wineQuality["type"].value_counts())
# Converting the value of ['type'] from 'white' to 0, 'red' to 1
type_mapping = {'white': 0, 'red': 1}
wineQuality['type'] = wineQuality['type'].map(type_mapping)
print("\nAfter the categorical feature ['type'] :")
print(wineQuality["type"].value_counts())

features = wineQuality.drop(['quality'], axis=1)
target = wineQuality['quality']
features_train, features_test, target_train, target_test = \
    train_test_split(features, target, test_size=0.2, random_state=0)

print('\nThe data shape of features_train DataFrame:\n', features_train.shape)
print('\nThe data shape of features_test DataFrame:\n', features_test.shape)

# Find the correlation of quality and features
corr = wineQuality.corr()
print('\nThe correlation of quality and features:')
print(corr['quality'].sort_values(ascending=False), '\n')

# plot a heatmap to show the correlation
plt.subplots()
f1 = plt.figure()
sns.heatmap(corr, cmap="RdYlGn_r")
plt.show()
f1.savefig("wineQuality.pdf", bbox_inches='tight')

# plot the relationship between alcohol and quality
f2 = plt.figure()
plt.scatter(wineQuality['alcohol'], wineQuality['quality'], alpha=0.6)
plt.xlabel("alcohol")
plt.ylabel("quality")
plt.show()
f2.savefig("alcohol&quality.pdf", bbox_inches='tight')

# Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(features_train, target_train)
# Evaluate Naive Bayes model
score_NaiveBayes = round(gnb_model.score(features_train, target_train), 4)  # ???
print("Naive Bayes score:", "{0:.2%}".format(score_NaiveBayes))
target_pred = gnb_model.predict(features_test)
print('\nClassification report:')
print(classification_report(target_test, target_pred))

##KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features_train, target_train)
# Evaluate KNN model
score_knn = round(knn.score(features_train, target_train), 4)
print("KNN score:", "{0:.2%}".format(score_knn))
target_pred = knn.predict(features_test)
print('\nClassification report:')
print(classification_report(target_test, target_pred))

# SVM (Support Vector Machine) model
svc = SVC()
svc.fit(features_train, target_train)
score_svc = round(svc.score(features_train, target_train), 4)
print("svm accuracy is:", "{0:.2%}".format(score_svc))
# Evaluate SVM model
target_pred = svc.predict(features_test)
print('\nClassification report:')
print(classification_report(target_test, target_pred))
