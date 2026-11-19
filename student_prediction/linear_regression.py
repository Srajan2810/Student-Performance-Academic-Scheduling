
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
base_path = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    base_path / "data" / "student_performance.csv"
)

# 2. Select input variables (X)
X = df[
    [
        "Attendance",
        "Internal_Marks",
        "Assignment_Marks",
        "Study_Hours"
    ]
]

# 3. Select output variable (Y)
Y = df["Final_Marks"]

# 4. Divide data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 5. Create Linear Regression model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, Y_train)

# 7. Predict final marks
predictions = model.predict(X_test)

# 8. Evaluate model
mae = mean_absolute_error(Y_test, predictions)

mse = mean_squared_error(Y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(Y_test, predictions)

print("LINEAR REGRESSION RESULTS")
print("-------------------------")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# 9. Display model coefficients
print("\nMODEL COEFFICIENTS")

for name, value in zip(X.columns, model.coef_):
    print(name, ":", round(value, 3))

print("Intercept:", round(model.intercept_, 3))

# 10. Compare actual and predicted marks
comparison = pd.DataFrame({
    "Actual Marks": Y_test.values,
    "Predicted Marks": np.round(predictions, 2)
})

print("\nACTUAL VS PREDICTED MARKS")
print(comparison.head(10))

# 11. Plot actual vs predicted marks
plt.figure(figsize=(8, 6))

plt.scatter(
    Y_test,
    predictions,
    alpha=0.75
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")

plt.title("Linear Regression: Actual vs Predicted Marks")

plt.tight_layout()

plt.savefig(
    base_path / "graphs" / "regression_results.png",
    dpi=300
)

plt.show()

# 12. Predict marks for a new student
new_student = pd.DataFrame([{
    "Attendance": 85,
    "Internal_Marks": 78,
    "Assignment_Marks": 90,
    "Study_Hours": 4
}])

predicted_marks = model.predict(new_student)

print("\nNEW STUDENT PREDICTION")
print("Predicted Final Marks:", round(predicted_marks[0], 2))
