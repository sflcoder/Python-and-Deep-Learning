import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# reading data set
train_df = pd.read_csv('./glass.csv')
x_train = train_df.drop("Type",axis=1)
y_train = train_df["Type"]

# split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

cor = train_df.corr()
cor_target = abs(cor["Type"])
print(cor_target)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

# Select features according to correlation
x_train_selected =  x_train.drop(['RI', 'Si','K', 'Ca', 'Fe'],axis=1)
x_test_selected =x_test.drop(['RI', 'Si','K', 'Ca', 'Fe'],axis=1)


# train the model
linearSVM = LinearSVC(random_state=0,max_iter=1200000)
linearSVM.fit(x_train, y_train)

# accuracy score for train set and test set
accuracy_score_test = "{0:.2%}".format(linearSVM.score(x_test, y_test))
print('linear SVM accuracy on test set is:', accuracy_score_test)
accuracy_score_train = "{0:.2%}".format(linearSVM.score(x_train, y_train))
print('linear SVM accuracy on training set is: ', accuracy_score_train)
y_pred = linearSVM.predict(x_test)
print(classification_report(y_test, y_pred))

# accuracy score for train set and test set with selected features
linearSVM_selected = LinearSVC(random_state=0,max_iter=1200000)
linearSVM_selected.fit(x_train_selected, y_train)
accuracy_score_test = "{0:.2%}".format(linearSVM_selected.score(x_test_selected, y_test))
print('linear SVM accuracy with selected features on test set is:', accuracy_score_test)
accuracy_score_train = "{0:.2%}".format(linearSVM_selected.score(x_train_selected, y_train))
print('linear SVM accuracy with selected features on training set is: ', accuracy_score_train)
y_prep_select = linearSVM_selected.predict(x_test_selected)
print(classification_report(y_test, y_prep_select))


