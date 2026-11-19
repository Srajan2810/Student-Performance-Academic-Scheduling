
import pandas as pd
import numpy as np
from pathlib import Path

# Generate reproducible random data
np.random.seed(42)

# Number of students
n = 100

# Generate student information
student_ids = [f"S{i:03d}" for i in range(1, n + 1)]

attendance = np.random.randint(45, 101, n)
internal_marks = np.random.randint(30, 101, n)
assignment_marks = np.random.randint(35, 101, n)
study_hours = np.round(np.random.uniform(0.5, 6, n), 1)

# Generate final marks using a mathematical model
noise = np.random.normal(0, 7, n)

final_marks = (
    0.15 * attendance
    + 0.45 * internal_marks
    + 0.25 * assignment_marks
    + 2 * study_hours
    + noise
)

# Keep marks between 0 and 100
final_marks = np.clip(final_marks, 0, 100)
final_marks = np.round(final_marks, 1)

# Create a DataFrame
data = {
    "Student_ID": student_ids,
    "Attendance": attendance,
    "Internal_Marks": internal_marks,
    "Assignment_Marks": assignment_marks,
    "Study_Hours": study_hours,
    "Final_Marks": final_marks
}

df = pd.DataFrame(data)

# Save dataset inside the data folder
output_path = Path(__file__).resolve().parent.parent / "data" / "student_performance.csv"
df.to_csv(output_path, index=False)

# Display results
print("Dataset created successfully!")
print("Total students:", len(df))

print("\nFirst 10 students:")
print(df.head(10))

print("\nDataset information:")
print(df.describe())

print("\nFile saved at:", output_path)
