
import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-3, 3, 400)

# Convex function
y1 = x**2

# Non-convex function
y2 = x**4 - 4*x**2

# Plot convex function
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label="f(x) = x²")
plt.title("Convex Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
plt.savefig("graphs/convex_function.png")
plt.show()

# Plot non-convex function
plt.figure(figsize=(8, 5))
plt.plot(x, y2, label="f(x) = x⁴ - 4x²")
plt.title("Non-Convex Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
plt.savefig("graphs/non_convex_function.png")
plt.show()
