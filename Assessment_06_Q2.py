dhgfj import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([28, 23, 32, 35, 29, 30, 27, 34, 32]).reshape((-1, 1))
y = np.array([403, 61, 634, 570, 298, 626, 447, 612, 258])

##### MODEL #####
model = LinearRegression().fit(x, y)

##### COEFFICIENTS #####
print("************ COEFFICIENTS ************")
a = model.intercept_
print(f"a: {model.intercept_}")

b = model.coef_
print(f"b: {model.coef_}")

##### COEFFICIENTS #####
temperature = 27
prediction = a + b*temperature

print(print("************ GUESTS EXPECTED ************"))
print(f'The Guests expected are: {prediction}')
