import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio. templates

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

load_boston = load_boston()
x = load_boston.data
y = load_boston.target

data = pd. DataFrame(x, columns=load_boston.feature_name)
data["Sale Price"]
data.head()
print(load_boston.DESCR)
print(data.shape)
print(data.describe)

#EDA
data.isnull.sum()
sns.pairplot(data, height=2.5)
sns.tight_layout()
sns.distplot(data["Sale Price"]);

print("Skewness:%f" %data["Sale Price"]. skew())
print("Kurtosis:%f" %data["Sale Price"]. kurt())

fig, ax = plt.subplots()
ax.scatter(x=data['CRIM'], y=data['Sale Price'])
plt.ylabel('Sale Price', fontsize=13) 
plt.xlabel('CRIM', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x=data['AGE'], y=data['Sale Price'])
plt.ylabel('Sale Price', fontsize=13) 
plt.xlabel('AGE', fontsize=13)
plt.show()

from scipy import stats
from scipy.stats import norm, skew

sns.distplot(data['Sale Price'], fit=norm) 
(mu, sigma) = norm.fit(data['Sale Price'])

print('\n mu ={:.2f} and sigma{:.2f}\n'.format(mu, sigma))
plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Sale Price Distribution')

#Get also QQ Plot Qurtile
fig = plt.figure()
res = stats.probplot(data['Sale Price'], plot=plt)
plt.show()

# Now
data['Sale Price'] = np.log1p(data['Sale Price'])
sns.distplot(data['Sale Price'], fit=norm) 
(mu, sigma) = norm.fit(data['Sale Price'])

print('\n mu ={:.2f} and sigma{:.2f}\n'.format(mu, sigma))
plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Sale Price Distribution')

#Get also QQ Plot Qurtile
fig = plt.figure()
res = stats.probplot(data['Sale Price'], plot=plt)
plt.show()

# Data Correlation
plt.figure(figsize=(10,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
plt.show()

# Absolute values of correlation
cor_target = abs(cor['Sale Price'])
# Get high value of Correlation
relevant_features cor_target[cor_target > 0.2]
# Getting the name of feature
names = [index for index, value in relevant_features.iteritems()]
# Remove target name
names.remove('Sale Price')
# Print features
print(names)
print(len(names))

# Model Building
from sklearn.model_selection import train_test_split

x = data.drop("Sale Price", axis=1)
y = data['Sale Price']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=42)

print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(xtrain, ytrain)

predictions = lr.predict(xtest)

print("Actual value of the house: " ytest[0])
print("Model Predicted Value: " predictions[0])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(ytest, predictions)
rmse = np.sqrt(mse)

print(mse)
print(rmse)
