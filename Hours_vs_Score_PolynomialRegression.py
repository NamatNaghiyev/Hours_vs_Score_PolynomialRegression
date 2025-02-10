import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create the dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [2, 4, 9, 15, 25, 40, 60, 85, 115, 150]
}

df = pd.DataFrame(data)

# Define X and y
X = df[['Hours']]
y = df['Score']

# Apply polynomial transformation
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Generate values for plotting
x_values = np.linspace(0, 11, 100).reshape(-1, 1)
x_values_poly = poly.transform(x_values)
y_values = model.predict(x_values_poly)

# Plot the results
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(x_values, y_values, color='blue', label='Polynomial Regression')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()
