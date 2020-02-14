import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# reading data set
train_df = pd.read_csv('./glass.csv')
x_train = train_df.drop("Type",axis=1)
y_train = train_df["Type"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(x_train, y_train)

print ('The accuracy of Naive Bayes(GaussianNB) Classifier is',"{0:.2%}".format(gnb.score(x_test,y_test)))
y_pred = gnb.fit(x_train, y_train).predict(x_test)

mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_pre = mnb.predict((x_test))


print ('The accuracy of Naive Bayes(MultinomialNB) Classifier is',"{0:.2%}".format(mnb.score(x_test,y_test)))
print(classification_report(y_test, y_pred))