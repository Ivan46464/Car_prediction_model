import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



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


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)
lin_reg = LinearRegression()
sfs = SFS(lin_reg, k_features=4,
           forward=True ,
           floating=False,
           scoring='r2',
           cv=0)
sfs.fit(X_scaled,y)
print(sfs.subsets_)
print(sfs.subsets_[3]['avg_score'])
plot_sfs(sfs.get_metric_dict())
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.2,random_state=42)
lin_reg.fit(X_train, y_train)
print(f"Score:{lin_reg.score(X_test,y_test)}")
y_pred_poly = lin_reg.predict(X_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Degree: {2}")
print(f"Mean Squared Error (Polynomial Regression): {mse_poly}")
print(f"R-squared (Polynomial Regression): {r2_poly}")

