
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

# Exams and available time slots
exams = ["Mathematics", "Programming", "Statistics"]
slots = ["Morning", "Afternoon"]

n_exams = len(exams)
n_slots = len(slots)
n_variables = n_exams * n_slots

# Variable order:
# Math-Morning, Math-Afternoon,
# Programming-Morning, Programming-Afternoon,
# Statistics-Morning, Statistics-Afternoon

# Objective: prefer morning slots when possible
c = np.array([0, 1, 0, 1, 0, 1], dtype=float)

rows = []
lower_bounds = []
upper_bounds = []

# Constraint 1: Every exam must be assigned exactly once
for i in range(n_exams):
    row = np.zeros(n_variables)

    for j in range(n_slots):
        row[i * n_slots + j] = 1

    rows.append(row)
    lower_bounds.append(1)
    upper_bounds.append(1)

# Conflicting exam pairs
conflicts = [
    (0, 1),  # Mathematics and Programming
    (1, 2)   # Programming and Statistics
]

# Constraint 2: Conflicting exams cannot share a slot
for exam_a, exam_b in conflicts:
    for j in range(n_slots):
        row = np.zeros(n_variables)

        row[exam_a * n_slots + j] = 1
        row[exam_b * n_slots + j] = 1

        rows.append(row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(1)

# Build constraint matrix
A = np.array(rows)

constraints = LinearConstraint(
    A,
    np.array(lower_bounds),
    np.array(upper_bounds)
)

# Variables must be binary: 0 or 1
bounds = Bounds(
    np.zeros(n_variables),
    np.ones(n_variables)
)

integrality = np.ones(n_variables)

# Solve using Mixed Integer Linear Programming
result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints
)

if result.success:
    print("Optimal timetable found!\n")

    timetable = []

    for i in range(n_exams):
        for j in range(n_slots):
            value = result.x[i * n_slots + j]

            if value > 0.5:
                timetable.append({
                    "Exam": exams[i],
                    "Time Slot": slots[j]
                })

    df = pd.DataFrame(timetable)

    print(df.to_string(index=False))

    df.to_csv(
        "exam_scheduling/exam_timetable.csv",
        index=False
    )

    print("\nTimetable saved successfully!")

else:
    print("No feasible timetable found.")
    print(result.message)
