import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 📌 Create the dataset: Study hours vs. score
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of hours studied
    'Score': [2, 4, 9, 15, 25, 40, 60, 85, 115, 150]  # Score obtained
}

df = pd.DataFrame(data)  # Convert dictionary to DataFrame

# 📌 Define features (X) and target variable (y)
X = df[['Hours']]  # Independent variable (study hours)
y = df['Score']    # Dependent variable (score)

# 📌 Apply polynomial transformation (degree = 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)  # Transform X into polynomial features

# 📌 Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# 📌 Generate values for plotting the polynomial curve
x_values = np.linspace(0, 11, 100).reshape(-1, 1)  # Generate 100 values from 0 to 11
x_values_poly = poly.transform(x_values)  # Transform the values using polynomial features
y_values = model.predict(x_values_poly)  # Predict scores using the trained model

# 📌 Plot the results
plt.scatter(X, y, color='red', label='Actual Data')  # Scatter plot of real data points
plt.plot(x_values, y_values, color='blue', label='Polynomial Regression')  # Regression curve

plt.xlabel('Hours')   # X-axis label
plt.ylabel('Score')   # Y-axis label
plt.legend()          # Show legend
plt.tight_layout()    # Adjust layout for better visualization
plt.show()            # Display the plot

