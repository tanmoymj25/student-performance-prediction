import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("StudentPerformanceFactors.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# ==========================================
# 3. IDENTIFY NUMERICAL & CATEGORICAL COLUMNS
# ==========================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ==========================================
# 4. NUMERICAL PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 5. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 6. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 8. LINEAR REGRESSION
# ==========================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)


# ==========================================
# 9. DECISION TREE
# ==========================================

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

decision_tree_model.fit(X_train, y_train)

tree_prediction = decision_tree_model.predict(X_test)


# ==========================================
# 10. LINEAR REGRESSION EVALUATION
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_rmse = mean_squared_error(
    y_test,
    linear_prediction
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


# ==========================================
# 11. DECISION TREE EVALUATION
# ==========================================

tree_mae = mean_absolute_error(
    y_test,
    tree_prediction
)

tree_rmse = mean_squared_error(
    y_test,
    tree_prediction
) ** 0.5

tree_r2 = r2_score(
    y_test,
    tree_prediction
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print("        MODEL COMPARISON")
print("========================================")

print("\nLinear Regression")
print("-------------------------")
print("MAE :", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2  :", round(linear_r2, 4))

print("\nDecision Tree")
print("-------------------------")
print("MAE :", round(tree_mae, 2))
print("RMSE:", round(tree_rmse, 2))
print("R2  :", round(tree_r2, 4))


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

if linear_r2 >= tree_r2:
    best_model = linear_model
    best_model_name = "Linear Regression"
    best_r2 = linear_r2

else:
    best_model = decision_tree_model
    best_model_name = "Decision Tree"
    best_r2 = tree_r2


print("\n========================================")
print("           BEST MODEL")
print("========================================")

print("Model:", best_model_name)
print("R2 Score:", round(best_r2, 4))


# ==========================================
# 14. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "best_model.joblib"
)

print("\nBest model saved successfully!")
print("File: best_model.joblib")
