
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Create pass/fail labels
df["Result"] = (df["Final_Marks"] >= 50).astype(int)

# Features and target
X = df[[
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Study_Hours"
]]

y = df["Result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Reduce four features to two dimensions for visualization
pca = PCA(n_components=2)
X_visual = pca.fit_transform(X_train)

# Plot the classes
plt.figure(figsize=(8, 5))
plt.scatter(
    X_visual[:, 0],
    X_visual[:, 1],
    c=y_train,
    cmap="coolwarm"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Student Classification Data")
plt.colorbar(label="Result (0 = Fail, 1 = Pass)")
plt.grid()

plt.savefig("graphs/svm_classification.png")
plt.show()
