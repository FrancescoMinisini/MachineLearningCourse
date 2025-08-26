import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

n = 100
x = np.random.rand(n, 1)
y = 2.0 + 5 * x**2 + 0.1 * np.random.randn(n, 1)


line_model = LinearRegression().fit(x, y)
line_predict = line_model.predict(x)

line_mse = mean_squared_error(y_true=y, y_pred=line_predict)



poly_features = PolynomialFeatures(degree=2).fit_transform(X=x)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2)
poly_model = LinearRegression().fit(X_train, y_train)
poly_predict = poly_model.predict(X_test)
poly_mse = mean_squared_error(y_true=y_test, y_pred=poly_predict)

print(poly_mse)

# print(line_model.coef_)
# print(line_model.intercept_)
# print(line_model.get_params().items())
plt.scatter(x, y, label = "Data")
plt.scatter(X_test[:,1], poly_predict, label = "Poly model")
# plt.scatter(x, y, label = "Data")
# plt.scatter(x, line_predict, label = "Line model")
plt.legend()
plt.show()