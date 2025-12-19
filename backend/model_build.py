"""
model_build.py

End-to-end ML pipeline construction, training, evaluation & persistence.
NO data leakage. Pipeline-safe.
"""

from typing import Dict, List, Optional, Tuple, Any
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from utils import validate_required_columns, convert_yes_no_columns


# Feature Definition


FEATURE_COLUMNS = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

TARGET_COLUMN = "ProductPitched"



# Preprocessor


def build_preprocessor(
    numeric_cols: List[str],
    ordinal_dict: Dict[str, List[str]],
    ohe_cols: List[str]
) -> ColumnTransformer:

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=list(ordinal_dict.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("ord", ord_pipe, list(ordinal_dict.keys())),
        ("ohe", ohe_pipe, ohe_cols)
    ])


# Pipeline Builder


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])



# Train & Evaluate


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    test_size: float = 0.2,
    grid: Optional[Dict[str, List[Any]]] = None
) -> Tuple[Pipeline, Dict[str, Any]]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=42
    )

    if grid:
        search = GridSearchCV(
            pipeline, grid, cv=5,
            scoring="f1_weighted",
            n_jobs=-1, verbose=2
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = pipeline.fit(X_train, y_train)
        best_params = None

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": best_params
    }

    return model, metrics


def save_pipeline(pipeline: Pipeline, path: str):
    joblib.dump(pipeline, path)
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Train Product Pitch Recommendation Model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="product_pitch_pipeline.pkl")
    args = parser.parse_args()

    
    # Load data
    
    df = pd.read_csv(args.data)

    is_valid, missing = validate_required_columns(
        df, FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing}")

    
    # Feature / Target split
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Binary normalization (safe)
    X = convert_yes_no_columns(X, ["Passport", "OwnCar"])

  
    # Column grouping
  
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    ordinal_dict = {
        "Occupation": ['Free Lancer', 'Salaried', 'Small Business', 'Large Business'],
        "Designation": ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
    }

    ohe_cols = [
        "TypeofContact", "Gender", "MaritalStatus"
    ]

    
    # Build pipeline
    
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        ordinal_dict=ordinal_dict,
        ohe_cols=ohe_cols
    )

    pipeline = build_pipeline(preprocessor)

    
    # Train
    
    model, metrics = train_and_evaluate(
        X, y, pipeline,
        grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    )

    
    # Results
    
    print("\n📊 MODEL PERFORMANCE")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    print("\n📌 Best Parameters:")
    print(metrics["best_params"])

    
    # Save model
    
    save_pipeline(model, args.output)
    print(f"\n Model saved to: {args.output}")
