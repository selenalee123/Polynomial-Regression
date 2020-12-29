
#%%
#import the libraries
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
#%%
##Print out x values
print(X)
##Print out y values
print(y)


#%%
#import train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test, = train_test_split(X,y, test_size=0.2, random_state =0)

#%%
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#%%
print(X)
print(y)
# %%
#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


#%%
y_pred= regressor.predict([[6.5]])
y_pred= sc_y.inverse_transform(y_pred)

#%%
# Visualizing the SVR result
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X), color="green")
plt.title("True or Bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#%%
#
#Visualizing the Polynomial Regression results ( for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid), color="blue")
plt.title("True or Bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# %%
