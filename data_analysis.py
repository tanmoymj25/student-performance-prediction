import pandas as pd

# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Exam Score Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Exam_Score", kde=True)

plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()


# -----------------------------
# 2. Hours Studied vs Exam Score
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Hours_Studied",
    y="Exam_Score"
)

plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()


# -----------------------------
# 3. Attendance vs Exam Score
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Attendance",
    y="Exam_Score"
)

plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()


# -----------------------------
# 4. Previous Scores vs Exam Score
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Previous_Scores",
    y="Exam_Score"
)

plt.title("Previous Scores vs Exam Score")
plt.xlabel("Previous Scores")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()