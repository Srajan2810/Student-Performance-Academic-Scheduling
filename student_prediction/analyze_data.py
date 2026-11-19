
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate and read the dataset
base_path = Path(__file__).resolve().parent.parent

data_path = base_path / "data" / "student_performance.csv"

df = pd.read_csv(data_path)

# Display basic information
print("TOTAL STUDENTS:", len(df))

print("\nFIRST 5 STUDENTS:")
print(df.head())

# Calculate statistical measures
print("\nAVERAGE VALUES:")
print(df.mean(numeric_only=True))

print("\nMEDIAN VALUES:")
print(df.median(numeric_only=True))

print("\nSTANDARD DEVIATION:")
print(df.std(numeric_only=True))

# Calculate correlation matrix
correlation = df.select_dtypes(include="number").corr()

print("\nCORRELATION MATRIX:")
print(correlation.round(2))

# Find correlation with final marks
print("\nCORRELATION WITH FINAL MARKS:")
print(correlation["Final_Marks"].sort_values(ascending=False))

# Create correlation heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Student Academic Factors")
plt.tight_layout()

graph_path = base_path / "graphs" / "correlation_heatmap.png"
plt.savefig(graph_path, dpi=300)
plt.show()

print("\nHeatmap saved successfully!")
