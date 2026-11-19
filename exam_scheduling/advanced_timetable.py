
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

# Exams and student counts
exams = ["Mathematics", "Programming", "Statistics"]
students = [40, 60, 30]

# Rooms and capacities
rooms = ["Room A", "Room B"]
capacities = [50, 70]

# Time slots
slots = ["Morning", "Afternoon"]

E = len(exams)
R = len(rooms)
T = len(slots)

# One binary variable for every exam-room-slot combination
def index(e, r, t):
    return e * R * T + r * T + t

n = E * R * T

# Objective: prefer morning slots
c = np.zeros(n)

for e in range(E):
    for r in range(R):
        c[index(e, r, 0)] = 0
        c[index(e, r, 1)] = 1

rows = []
lower = []
upper = []

# Every exam must be scheduled exactly once
for e in range(E):
    row = np.zeros(n)

    for r in range(R):
        for t in range(T):
            row[index(e, r, t)] = 1

    rows.append(row)
    lower.append(1)
    upper.append(1)

# Room capacity constraints
# An exam can only use a room if capacity is sufficient
for e in range(E):
    for r in range(R):
        if students[e] > capacities[r]:
            row = np.zeros(n)

            for t in range(T):
                row[index(e, r, t)] = 1

            rows.append(row)
            lower.append(0)
            upper.append(0)

# A room can host at most one exam in a time slot
for r in range(R):
    for t in range(T):
        row = np.zeros(n)

        for e in range(E):
            row[index(e, r, t)] = 1

        rows.append(row)
        lower.append(0)
        upper.append(1)

# Student conflict pairs
conflicts = [
    (0, 1),  # Mathematics and Programming
    (1, 2)   # Programming and Statistics
]

# Conflicting exams cannot be scheduled at the same time
for e1, e2 in conflicts:
    for t in range(T):
        row = np.zeros(n)

        for r in range(R):
            row[index(e1, r, t)] = 1
            row[index(e2, r, t)] = 1

        rows.append(row)
        lower.append(0)
        upper.append(1)

# Solve optimization problem
A = np.array(rows)

constraints = LinearConstraint(
    A,
    np.array(lower),
    np.array(upper)
)

bounds = Bounds(np.zeros(n), np.ones(n))
integrality = np.ones(n)

result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints
)

if result.success:
    timetable = []

    for e in range(E):
        for r in range(R):
            for t in range(T):

                if result.x[index(e, r, t)] > 0.5:
                    timetable.append({
                        "Exam": exams[e],
                        "Students": students[e],
                        "Room": rooms[r],
                        "Time": slots[t]
                    })

    df = pd.DataFrame(timetable)

    print("\nOPTIMIZED EXAM TIMETABLE\n")
    print(df.to_string(index=False))

    df.to_csv(
        "exam_scheduling/advanced_timetable.csv",
        index=False
    )

    print("\nTimetable saved successfully!")

else:
    print("No feasible timetable found.")
    print(result.message)
