import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('winequality-red.csv')
# handling missing value
data = train.dropna()
print(sum(data.isnull().sum() != 0))

# find the top 3 most correlated features to the target label(quality)
corr = train.corr()
print(corr['quality'].sort_values(ascending=False)[:3], '\n')
# plt.hist(train.quality)
# plt.show()

##Build a linear model
# Log transform the target
y = np.log(train.quality)
plt.hist(y)
plt.show()
X = data[['quality', 'alcohol', 'sulphates']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('RMSE is: \n', mean_squared_error(y_test, predictions))
