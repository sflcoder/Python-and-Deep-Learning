from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# Using Naive Bayes
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using Naive Bayes: ", score)

# Using SVM
classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, twenty_train.target)
predicted = classifier.predict(X_test_tfidf)

score_svm = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using SVM: ", score_svm)

# b
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using Naive Bayes: ", score)

# Using SVM

classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, twenty_train.target)
predicted = classifier.predict(X_test_tfidf)

score_svm = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using SVM: ", score_svm)

# c
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using Naive Bayes: ", score)

# Using SVM

classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, twenty_train.target)
predicted = classifier.predict(X_test_tfidf)

score_svm = metrics.accuracy_score(twenty_test.target, predicted)
print("Score using SVM: ", score_svm)
