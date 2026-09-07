import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("StudentPerformanceFactors.csv")

print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# ==========================================
# 3. Identify Numerical and Categorical Columns
# ==========================================

numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ==========================================
# 4. Numerical Preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 5. Categorical Preprocessing
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 6. Combine Both Pipelines
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ==========================================
# 7. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 8. Fit Preprocessor
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)


# ==========================================
# 9. Save Preprocessor
# ==========================================

joblib.dump(preprocessor, "preprocessor.joblib")

print("\nPreprocessor saved successfully!")