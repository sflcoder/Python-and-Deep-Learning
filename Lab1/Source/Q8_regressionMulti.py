import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns; sns.set(color_codes=True)

x = pd.read_csv('Fish.csv')

# Finding null values
nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Interpolating the null Values
data = x.select_dtypes(include=[np.number]).interpolate().dropna()


# Plotting correlation for the data
plt.figure(figsize=(20,20))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap="YlGnBu")
plt.show()

# Printing the correlation with target 'Species'
print(cor['Species'].sort_values(ascending=False)[:5],'\n')

# Build a multiple linear regression model
y = data['Species']
X = data.drop(['Species'],axis =1)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.20)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

# Visualise the Predicted vs Actual
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='r')
# alpha shows overlapping data
plt.xlabel('Predicted ')
plt.ylabel('Actual')
plt.title('Linear Regression Model')
plt.show()