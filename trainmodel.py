# ============================================================
# STUDENT PERFORMANCE PREDICTION - ML MODEL
# Research-based implementation
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "student-mat.csv"

# True  = predict without G1 and G2
# False = use G1 and G2 as previous academic information
EARLY_PREDICTION = False


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    print("\nERROR:")
    print(f"Dataset not found: {DATA_PATH}")
    print("\nMake sure student-mat.csv is inside:")
    print(os.getcwd())

    exit()


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=";"
)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

# Remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Remove accidental spaces inside column names
df.columns = df.columns.str.replace(" ", "", regex=False)

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 5. CHECK GRADE COLUMNS
# ============================================================

print("\nChecking grade columns...")

if "G1" in df.columns:
    print("G1 column found.")

if "G2" in df.columns:
    print("G2 column found.")

if "G3" in df.columns:
    print("G3 column found.")


# ============================================================
# 6. FIND G1 / G2 / G3 IF COLUMN NAMES ARE DIFFERENT
# ============================================================

# Convert column names to uppercase for comparison

column_map = {
    col.upper(): col
    for col in df.columns
}


# Find G1
if "G1" not in df.columns:

    if "G1" in column_map:

        actual_name = column_map["G1"]

        df.rename(
            columns={actual_name: "G1"},
            inplace=True
        )

        print(f"G1 renamed from: {actual_name}")


# Find G2
if "G2" not in df.columns:

    if "G2" in column_map:

        actual_name = column_map["G2"]

        df.rename(
            columns={actual_name: "G2"},
            inplace=True
        )

        print(f"G2 renamed from: {actual_name}")


# Find G3
if "G3" not in df.columns:

    if "G3" in column_map:

        actual_name = column_map["G3"]

        df.rename(
            columns={actual_name: "G3"},
            inplace=True
        )

        print(f"G3 renamed from: {actual_name}")


# ============================================================
# 7. VERIFY TARGET COLUMN
# ============================================================

if "G3" not in df.columns:

    print("\nERROR: G3 column was not found.")

    print("\nYour dataset contains these columns:")
    print(df.columns.tolist())

    print(
        "\nThe model needs a final-grade column (G3) "
        "to create the prediction target."
    )

    exit()


# ============================================================
# 8. CONVERT GRADE COLUMNS TO NUMERIC
# ============================================================

df["G3"] = pd.to_numeric(
    df["G3"],
    errors="coerce"
)

if "G1" in df.columns:

    df["G1"] = pd.to_numeric(
        df["G1"],
        errors="coerce"
    )


if "G2" in df.columns:

    df["G2"] = pd.to_numeric(
        df["G2"],
        errors="coerce"
    )


# ============================================================
# 9. REMOVE INVALID G3 ROWS
# ============================================================

df = df.dropna(
    subset=["G3"]
)

print("\nDataset after cleaning:", df.shape)


# ============================================================
# 10. CREATE PERFORMANCE TARGET
# ============================================================

# G3 >= 10  → Pass
# G3 < 10   → Fail

df["performance"] = np.where(
    df["G3"] >= 10,
    "Pass",
    "Fail"
)


print("\nPerformance distribution:")
print(
    df["performance"].value_counts()
)


# ============================================================
# 11. CREATE G1 + G2 FEATURE
# ============================================================

# Only create this feature if both columns exist.

if "G1" in df.columns and "G2" in df.columns:

    df["G1_G2_sum"] = (
        df["G1"] + df["G2"]
    )

    df["G1_G2_average"] = (
        df["G1"] + df["G2"]
    ) / 2

    print("\nG1 + G2 features created.")


else:

    print(
        "\nG1 or G2 not available."
    )

    print(
        "Skipping G1/G2 engineered features."
    )


# ============================================================
# 12. CREATE X AND y
# ============================================================

# Remove final grade G3 because G3 is what we predict.

X = df.drop(
    columns=[
        "G3",
        "performance"
    ],
    errors="ignore"
)

y = df["performance"]


# ============================================================
# 13. EARLY PREDICTION OPTION
# ============================================================

if EARLY_PREDICTION:

    print("\nEARLY PREDICTION MODE")
    print("---------------------")

    # Do not allow previous grades to be used.

    X = X.drop(
        columns=[
            "G1",
            "G2",
            "G1_G2_sum",
            "G1_G2_average"
        ],
        errors="ignore"
    )

    print("G1 removed.")
    print("G2 removed.")
    print("G1_G2_sum removed.")
    print("G1_G2_average removed.")

else:

    print("\nPREVIOUS-GRADE MODE")
    print("-------------------")

    if "G1" in X.columns:
        print("G1 included.")

    if "G2" in X.columns:
        print("G2 included.")

    if "G1_G2_sum" in X.columns:
        print("G1_G2_sum included.")

    if "G1_G2_average" in X.columns:
        print("G1_G2_average included.")


# ============================================================
# 14. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)


print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# 15. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


# ============================================================
# 16. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


# ============================================================
# 17. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )

    ]
)


# ============================================================
# 18. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 19. DEFINE ML MODELS
# ============================================================

models = {

    "Gradient Boosting":
        GradientBoostingClassifier(

            n_estimators=150,

            learning_rate=0.05,

            max_depth=3,

            random_state=42
        ),


    "Extra Trees":
        ExtraTreesClassifier(

            n_estimators=300,

            random_state=42,

            class_weight="balanced"
        ),


    "Logistic Regression":
        LogisticRegression(

            max_iter=2000,

            class_weight="balanced",

            random_state=42
        )
}


# ============================================================
# 20. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


results = {}


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


for model_name, model in models.items():

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]
    )


    scores = cross_validate(

        pipeline,

        X_train,

        y_train,

        cv=cv,

        scoring=[
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted"
        ],

        n_jobs=-1
    )


    results[model_name] = {

        "Accuracy":
            scores["test_accuracy"].mean(),

        "Precision":
            scores["test_precision_weighted"].mean(),

        "Recall":
            scores["test_recall_weighted"].mean(),

        "F1":
            scores["test_f1_weighted"].mean()
    }


    print("\n" + model_name)

    print(
        "Accuracy :",
        round(
            results[model_name]["Accuracy"],
            4
        )
    )

    print(
        "Precision:",
        round(
            results[model_name]["Precision"],
            4
        )
    )

    print(
        "Recall   :",
        round(
            results[model_name]["Recall"],
            4
        )
    )

    print(
        "F1 Score :",
        round(
            results[model_name]["F1"],
            4
        )
    )


# ============================================================
# 21. SELECT BEST MODEL
# ============================================================

best_model_name = max(

    results,

    key=lambda name:
        results[name]["F1"]
)


print("\n")
print("=" * 70)

print(
    "SELECTED MODEL:",
    best_model_name
)

print("=" * 70)


# ============================================================
# 22. TRAIN FINAL MODEL
# ============================================================

best_model = models[
    best_model_name
]


final_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            best_model
        )

    ]
)


final_pipeline.fit(

    X_train,

    y_train
)


# ============================================================
# 23. TEST MODEL
# ============================================================

y_pred = final_pipeline.predict(
    X_test
)


# ============================================================
# 24. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(

    y_test,

    y_pred,

    average="weighted"
)


recall = recall_score(

    y_test,

    y_pred,

    average="weighted"
)


f1 = f1_score(

    y_test,

    y_pred,

    average="weighted"
)


# ============================================================
# 25. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)

print("FINAL TEST RESULTS")

print("=" * 70)


print(
    "Accuracy :",
    round(accuracy * 100, 2),
    "%"
)


print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)


print(
    "Recall   :",
    round(recall * 100, 2),
    "%"
)


print(
    "F1 Score :",
    round(f1 * 100, 2),
    "%"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[
        "Fail",
        "Pass"
    ]
)


plt.figure(
    figsize=(6, 5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=[
        "Fail",
        "Pass"
    ],

    yticklabels=[
        "Fail",
        "Pass"
    ]
)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Student Performance Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    "confusion_matrix.png",

    dpi=300
)


plt.show()


# ============================================================
# 28. SAVE MODEL
# ============================================================

joblib.dump(

    final_pipeline,

    "student_model.pkl"
)


# ============================================================
# 29. SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "model_name":
        best_model_name,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1_score":
        f1,

    "early_prediction":
        EARLY_PREDICTION,

    "features":
        list(X.columns)
}


joblib.dump(

    model_info,

    "model_info.pkl"
)


# ============================================================
# 30. FINISHED
# ============================================================

print("\n")
print("=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)

print("\nSaved files:")

print("1. student_model.pkl")
print("2. model_info.pkl")
print("3. confusion_matrix.png")

print("\nDone.")