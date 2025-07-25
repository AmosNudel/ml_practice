from py_scripts import data_preprocessing_template as dpt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(dpt.X_train, dpt.y_train)

# Predicting the Test set results
y_pred = regressor.predict(dpt.X_test)

# Visualising the Training set results
plt.scatter(dpt.X_train, dpt.y_train, color = 'red')
plt.plot(dpt.X_train, regressor.predict(dpt.X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(dpt.X_test, dpt.y_test, color = 'red')
plt.plot(dpt.X_train, regressor.predict(dpt.X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()