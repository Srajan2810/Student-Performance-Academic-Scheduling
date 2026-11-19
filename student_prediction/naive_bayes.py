
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset
base_path = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    base_path / "data" / "student_performance.csv"
)

# 2. Create risk categories
def classify_risk(marks):
    if marks < 50:
        return "High Risk"
    elif marks < 70:
        return "Moderate Risk"
    else:
        return "Low Risk"

df["Risk_Category"] = df["Final_Marks"].apply(classify_risk)

# 3. Select input features
X = df[
    [
        "Attendance",
        "Internal_Marks",
        "Assignment_Marks",
        "Study_Hours"
    ]
]

# 4. Select target category
Y = df["Risk_Category"]

# 5. Display category distribution
print("STUDENT RISK DISTRIBUTION")
print(Y.value_counts())

# 6. Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# 7. Create Naive Bayes model
model = GaussianNB()

# 8. Train model
model.fit(X_train, Y_train)

# 9. Predict risk categories
predictions = model.predict(X_test)

# 10. Evaluate model
accuracy = accuracy_score(Y_test, predictions)

print("\nNAIVE BAYES RESULTS")
print("-------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")

print(classification_report(
    Y_test,
    predictions,
    zero_division=0
))

# 11. Create confusion matrix
labels = ["High Risk", "Moderate Risk", "Low Risk"]

cm = confusion_matrix(
    Y_test,
    predictions,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Naive Bayes Confusion Matrix")

plt.tight_layout()

plt.savefig(
    base_path / "graphs" / "naive_bayes_confusion.png",
    dpi=300
)

plt.show()

# 12. Predict a new student's category
new_student = pd.DataFrame([{
    "Attendance": 55,
    "Internal_Marks": 42,
    "Assignment_Marks": 48,
    "Study_Hours": 1.5
}])

predicted_category = model.predict(new_student)

probabilities = model.predict_proba(new_student)

print("\nNEW STUDENT PREDICTION")
print("Risk Category:", predicted_category[0])

print("\nCATEGORY PROBABILITIES")

for category, probability in zip(
    model.classes_,
    probabilities[0]
):
    print(category, ":", round(probability * 100, 2), "%")

# 13. Create pie chart
plt.figure(figsize=(7, 7))

df["Risk_Category"].value_counts().reindex(labels).plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    colors=["#D1495B", "#F6AE2D", "#2E86AB"]
)

plt.title("Distribution of Student Risk Categories")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    base_path / "graphs" / "student_risk_pie.png",
    dpi=300
)

plt.show()

print("\nGraphs saved successfully!")
