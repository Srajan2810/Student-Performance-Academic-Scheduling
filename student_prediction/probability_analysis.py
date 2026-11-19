
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import binom, poisson, norm

# Locate project folders
base_path = Path(__file__).resolve().parent.parent

# Load student dataset
df = pd.read_csv(
    base_path / "data" / "student_performance.csv"
)

# Create output folder if needed
graph_path = base_path / "graphs"
graph_path.mkdir(exist_ok=True)

# ==================================
# 1. BERNOULLI DISTRIBUTION
# ==================================

print("\n1. BERNOULLI DISTRIBUTION")

# Pass = marks >= 50
pass_probability = (df["Final_Marks"] >= 50).mean()

fail_probability = 1 - pass_probability

print("Probability of passing:", round(pass_probability, 3))
print("Probability of failing:", round(fail_probability, 3))

plt.figure(figsize=(6, 4))

plt.bar(
    ["Fail", "Pass"],
    [fail_probability, pass_probability],
    color=["tomato", "steelblue"]
)

plt.title("Bernoulli Distribution - Pass/Fail")
plt.ylabel("Probability")
plt.ylim(0, 1)

plt.savefig(graph_path / "bernoulli.png", dpi=300)
plt.show()


# ==================================
# 2. BINOMIAL DISTRIBUTION
# ==================================

print("\n2. BINOMIAL DISTRIBUTION")

n = 10
p = pass_probability

students = np.arange(0, n + 1)

probabilities = binom.pmf(students, n, p)

for k, probability in zip(students, probabilities):
    print(
        f"Probability of exactly {k} passes:",
        round(probability, 4)
    )

plt.figure(figsize=(8, 5))

plt.bar(students, probabilities, color="steelblue")

plt.title("Binomial Distribution - Number of Passing Students")
plt.xlabel("Number of students passing")
plt.ylabel("Probability")

plt.savefig(graph_path / "binomial.png", dpi=300)
plt.show()


# ==================================
# 3. POISSON DISTRIBUTION
# ==================================

print("\n3. POISSON DISTRIBUTION")

# Hypothetical average of 4 support requests per day
lam = 4

requests = np.arange(0, 13)

poisson_prob = poisson.pmf(requests, lam)

print("Average requests per day:", lam)
print(
    "Probability of exactly 6 requests:",
    round(poisson.pmf(6, lam), 4)
)

plt.figure(figsize=(8, 5))

plt.bar(requests, poisson_prob, color="orange")

plt.title("Poisson Distribution - Daily Support Requests")
plt.xlabel("Number of requests")
plt.ylabel("Probability")

plt.savefig(graph_path / "poisson.png", dpi=300)
plt.show()


# ==================================
# 4. NORMAL DISTRIBUTION
# ==================================

print("\n4. NORMAL DISTRIBUTION")

marks = df["Final_Marks"]

mean = marks.mean()
std = marks.std()

print("Mean marks:", round(mean, 2))
print("Standard deviation:", round(std, 2))

x = np.linspace(0, 100, 300)

y = norm.pdf(x, mean, std)

plt.figure(figsize=(8, 5))

plt.hist(
    marks,
    bins=10,
    density=True,
    alpha=0.6,
    color="steelblue",
    label="Student marks"
)

plt.plot(
    x,
    y,
    color="red",
    linewidth=2,
    label="Fitted Normal curve"
)

plt.title("Normal Distribution of Student Marks")
plt.xlabel("Final marks")
plt.ylabel("Density")
plt.legend()

plt.savefig(graph_path / "normal.png", dpi=300)
plt.show()


# ==================================
# 5. CENTRAL LIMIT THEOREM
# ==================================

print("\n5. CENTRAL LIMIT THEOREM")

population = marks.to_numpy()

sample_size = 30
number_of_samples = 1000

sample_means = []

for i in range(number_of_samples):

    sample = np.random.choice(
        population,
        size=sample_size,
        replace=True
    )

    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

print("Population mean:", round(mean, 2))

print(
    "Mean of sample means:",
    round(sample_means.mean(), 2)
)

print(
    "Standard error:",
    round(std / np.sqrt(sample_size), 2)
)

# Plot sample mean distribution
plt.figure(figsize=(8, 5))

plt.hist(
    sample_means,
    bins=30,
    density=True,
    alpha=0.6,
    color="steelblue",
    label="Sample means"
)

# Theoretical normal approximation
x_clt = np.linspace(
    sample_means.min(),
    sample_means.max(),
    300
)

y_clt = norm.pdf(
    x_clt,
    mean,
    std / np.sqrt(sample_size)
)

plt.plot(
    x_clt,
    y_clt,
    color="red",
    linewidth=2,
    label="Normal approximation"
)

plt.title("Central Limit Theorem Demonstration")
plt.xlabel("Sample mean")
plt.ylabel("Density")
plt.legend()

plt.savefig(graph_path / "clt.png", dpi=300)
plt.show()

print("\nAll probability graphs generated successfully!")
