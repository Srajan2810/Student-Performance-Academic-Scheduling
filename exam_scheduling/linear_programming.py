
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pathlib import Path

# Objective function:
# Maximize Z = 40x + 30y

# scipy.optimize.linprog minimizes,
# so we use negative coefficients.
c = [-40, -30]

# Constraints:
# 2x + y <= 100
# x + 2y <= 80

A = [
    [2, 1],
    [1, 2]
]

b = [100, 80]

# Decision variables must be non-negative
bounds = [(0, None), (0, None)]

# Solve using the dual Simplex algorithm
result = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=bounds,
    method="highs-ds"
)

# Display results
if result.success:

    x = result.x[0]
    y = result.x[1]

    maximum_benefit = -result.fun

    print("OPTIMAL SOLUTION FOUND")
    print("----------------------")

    print("Session A:", round(x, 2))
    print("Session B:", round(y, 2))

    print("Maximum Benefit:", round(maximum_benefit, 2))

else:
    print("No optimal solution found.")
    print(result.message)


# Draw the feasible region
x_values = np.linspace(0, 60, 300)

y1 = 100 - 2 * x_values
y2 = (80 - x_values) / 2

plt.figure(figsize=(8, 6))

plt.plot(x_values, y1, label="2x + y = 100")
plt.plot(x_values, y2, label="x + 2y = 80")

# Shade feasible region
plt.fill(
    [0, 50, 40, 0],
    [0, 0, 20, 40],
    alpha=0.25,
    label="Feasible Region"
)

# Mark optimal solution
plt.scatter(40, 20, color="red", s=100,
            label="Optimal Point (40, 20)")

plt.xlabel("Session A (x)")
plt.ylabel("Session B (y)")

plt.title("Linear Programming - Graphical Method")

plt.xlim(0, 60)
plt.ylim(0, 60)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

base_path = Path(__file__).resolve().parent.parent

plt.savefig(
    base_path / "graphs" / "linear_programming.png",
    dpi=300
)

plt.show()
