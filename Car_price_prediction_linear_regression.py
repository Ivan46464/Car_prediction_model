import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('CarPricesPrediction.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)
print(data.head())
print(data.columns)
print(data.isnull().sum())
print(data.describe())
print(data.info())
print(data['Make'].unique())
data['Make'] = data['Make'].replace({'Ford': 1, 'Toyota': 2, 'Chevrolet': 3, 'Nissan': 4, 'Honda': 5})
data['Make'] = data['Make'].astype(int)
print(data['Make'].head())
print(data['Model'].unique())
data['Model'] = data['Model'].replace({'Silverado':1, 'Civic':2, 'Altima':3, 'Camry':4, 'F-150':5})
data['Model'] = data['Model'].astype(int)
print(data['Model'].head())
data['Condition'] = data['Condition'].replace({'Fair':1, 'Good':2,'Excellent':3})
data['Condition'] = data['Condition'].astype(int)
print(data['Condition'].head())

numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()
lr = LinearRegression()
X = data[['Model', 'Year', 'Mileage', 'Condition']]
y = data['Price']

sfs = SFS(lr, k_features=4,
           forward=True ,
           floating=False,
           scoring='r2',
           cv=0)
sfs.fit(X,y)
print(sfs.subsets_)
print(sfs.subsets_[3]['avg_score'])
plot_sfs(sfs.get_metric_dict())
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr.fit(X_train,y_train)
print(f"Score:{lr.score(X_test,y_test)}")
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



