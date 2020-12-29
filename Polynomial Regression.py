#Polynoial Regression
#%%
#Import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
dataset.head(10)





#%%
#Print the value of X and y
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
print(X)
print(y)
#%%
##Split the data into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test, = train_test_split(X,y, test_size=0.2, random_state =0)

# %%
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#%%
##Fit the linear regression to the data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# %%
#Fit the Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#%%
#Visualising the linear regression results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X), color="green")
plt.title("True or Bluff(linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#%%
#Visualizing the Polynomial Regression results ( for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("True or Bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#%%
#Predict the new result with Linear Regression
lin_reg.predict([[6.5]])
#%%
#Predict a new result with Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
# %%

# %%
