import random

import pandas as pd
import numpy as np

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


random.seed(0)

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
X = data[['Model', 'Year', 'Mileage', 'Condition']].values
y = data['Price'].values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2,random_state=0)
ran_reg = RandomForestRegressor()
sfs = SFS(ran_reg, k_features=4,
           forward=True ,
           floating=False,
           scoring='r2',
           cv=0)
sfs.fit(X_scaled,y)
print(sfs.subsets_)
print(sfs.subsets_[3]['avg_score'])
plot_sfs(sfs.get_metric_dict())
plt.show()


parameters = {
    "max_depth": [n for n in range(1,20)],
    "n_estimators": [25,50, 100, 200,250,300],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
clf = GridSearchCV(estimator=ran_reg, param_grid=parameters, cv=5, verbose=2, n_jobs=-1)


clf.fit(x_train, y_train)


best_model = clf.best_estimator_
print(f"Best Model: {best_model}")
print(f"Best Parameters: {clf.best_params_}")


y_pred = best_model.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

print(f"Random Forest MSE: {mse_rf}")
print(f"Random Forest R-squared: {r2_rf}")