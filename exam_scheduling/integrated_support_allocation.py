
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.optimize import milp, Bounds, LinearConstraint

# 1. Load student dataset
df = pd.read_csv("data/student_performance.csv")

features = [
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Study_Hours"
]

X = df[features]
y = df["Final_Marks"]

# 2. Train prediction model
model = LinearRegression()
model.fit(X, y)

# 3. Predict student marks
df["Predicted_Marks"] = model.predict(X)

# 4. Identify students needing support
threshold = 60

at_risk = df[df["Predicted_Marks"] < threshold].copy()
at_risk = at_risk.reset_index(drop=True)

print("Students needing academic support:", len(at_risk))

if len(at_risk) == 0:
    print("No students identified for support.")
else:

    # 5. Define two support sessions
    slots = ["Morning", "Afternoon"]
    n_students = len(at_risk)
    n_slots = len(slots)

    # Each student-slot combination is a binary variable
    n_variables = n_students * n_slots

    def index(student, slot):
        return student * n_slots + slot

    # Priority: lower predicted marks = higher priority
    priority = threshold - at_risk["Predicted_Marks"].values

    # Minimize negative priority = maximize total priority
    c = np.zeros(n_variables)

    for i in range(n_students):
        for j in range(n_slots):
            c[index(i, j)] = -priority[i]

    rows = []
    lower = []
    upper = []

    # Each student can receive at most one session
    for i in range(n_students):
        row = np.zeros(n_variables)

        for j in range(n_slots):
            row[index(i, j)] = 1

        rows.append(row)
        lower.append(0)
        upper.append(1)

    # Each session can accommodate at most 10 students
    session_capacity = 10

    for j in range(n_slots):
        row = np.zeros(n_variables)

        for i in range(n_students):
            row[index(i, j)] = 1

        rows.append(row)
        lower.append(0)
        upper.append(session_capacity)

    # 6. Solve Integer Programming problem
    constraints = LinearConstraint(
        np.array(rows),
        np.array(lower),
        np.array(upper)
    )

    result = milp(
        c=c,
        integrality=np.ones(n_variables),
        bounds=Bounds(
            np.zeros(n_variables),
            np.ones(n_variables)
        ),
        constraints=constraints
    )

    # 7. Display and save allocation
    if result.success:
        allocation = []

        for i in range(n_students):
            for j in range(n_slots):

                if result.x[index(i, j)] > 0.5:
                    allocation.append({
                        "Student_ID": at_risk.loc[i, "Student_ID"],
                        "Predicted_Marks": round(
                            at_risk.loc[i, "Predicted_Marks"], 2
                        ),
                        "Priority": "High",
                        "Assigned_Session": slots[j]
                    })

        output = pd.DataFrame(allocation)

        print("\nOPTIMIZED SUPPORT ALLOCATION\n")

        if len(output) > 0:
            print(output.to_string(index=False))
        else:
            print("No support seats allocated.")

        output.to_csv(
            "exam_scheduling/support_allocation.csv",
            index=False
        )

        print("\nSaved support_allocation.csv")

    else:
        print("Optimization failed:", result.message)
