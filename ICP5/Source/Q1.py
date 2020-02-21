import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
# print(train.columns.values)
print(train.GarageArea.describe())
print(train.SalePrice.describe())
plt.scatter(train.GarageArea, train.SalePrice, s=75, alpha=.5)
plt.show()

# data_dropoutlier  = train[(train.GarageArea<1200)&(train.GarageArea>100)]
# select data based on the quantile
train = train[train.GarageArea.between(train.GarageArea.quantile(.15), train.GarageArea.quantile(.85))]
print(train.GarageArea.describe())
print(train.SalePrice.describe())
plt.scatter(train.GarageArea, train.SalePrice, s=75, alpha=.5)
plt.show()
