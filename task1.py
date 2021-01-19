import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head()

# Simple line plot 
x=data['Hours']
y=data['Scores']

# Plot
plt.scatter(x,y,c='green')

# Decorate
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Score/Marks')

plt.show()

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train) 

print("Model trained on dataset.")

print ('Coefficients: ', model.coef_)
print ('Intercept: ',model.intercept_)

# Plotting the regression line
l = model.coef_*x+model.intercept_

# Plotting for the test data
plt.title('Regression Line')
plt.scatter(x, y)
plt.plot(x, l);
plt.show()

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Hours': X_test[:,0], 'Actual Score': y_test, 'Predicted Score': y_pred})  
df

print('Number of hours: 9.25')
print('Predicted score: ', model.predict([[9.25]])[0])

from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred , y_test) )


