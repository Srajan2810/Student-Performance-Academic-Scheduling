
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load our existing student dataset
df = pd.read_csv("data/student_performance.csv")

# Use attendance to predict final marks
x = df["Attendance"].values
y = df["Final_Marks"].values

# Normalize attendance to make training more stable
x_mean = x.mean()
x_std = x.std()
x_scaled = (x - x_mean) / x_std

# Initialize parameters
m = 0.0
c = 0.0

learning_rate = 0.01
iterations = 1000
n = len(x)

loss_history = []

# Gradient Descent
for i in range(iterations):

    # Predictions
    y_pred = m * x_scaled + c

    # Calculate error
    error = y_pred - y

    # Calculate gradients
    dm = (2 / n) * np.sum(error * x_scaled)
    dc = (2 / n) * np.sum(error)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Calculate loss
    loss = np.mean(error ** 2)
    loss_history.append(loss)

print("Final slope:", m)
print("Final intercept:", c)
print("Final MSE:", loss_history[-1])

# Plot loss reduction
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent - Loss Reduction")
plt.grid()
plt.savefig("graphs/gradient_descent_loss.png")
plt.show()

# Plot actual vs predicted marks
plt.figure(figsize=(8, 5))
plt.scatter(x_scaled, y, label="Actual marks")
plt.plot(x_scaled, m * x_scaled + c,
         color="red", label="Predicted marks")
plt.xlabel("Normalized Attendance")
plt.ylabel("Final Marks")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid()
plt.savefig("graphs/gradient_descent_prediction.png")
plt.show()
