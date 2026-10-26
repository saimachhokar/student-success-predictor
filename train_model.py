"""Compare models that identify students who may be at risk of failing math."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Locate the dataset relative to this file so the script works from any folder.
DATA_FILE = Path(__file__).resolve().parent / "data" / "student-mat.csv"


def make_pipeline(classifier, preprocessor):
    """Combine preprocessing and a classifier to keep transformations in each fold."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    # The dataset uses semicolons between values rather than commas.
    data = pd.read_csv(DATA_FILE, sep=";")

    # A final grade below 10 means the student is at risk of failing.
    # Label at-risk students as 1 so precision, recall, and F1 focus on them.
    target = (data["G3"] < 10).astype(int)

    # Exclude all grades so no grade information can leak into the predictions.
    features = data.drop(columns=["G1", "G2", "G3"])

    # Keep the same at-risk/not-at-risk proportions in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    # Identify categorical and numeric columns for the appropriate transforms.
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns

    # Convert categories into numeric columns and scale numeric values. Dense
    # output also supports HistGradientBoostingClassifier.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )

    # Every candidate uses the same stratified folds and preprocessing pipeline.
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    scorers = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    # Compare all requested models, including a simple dummy baseline.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent",
            random_state=42,
        ),
    }

    print(
        "5-fold cross-validation results for the at-risk class "
        "(mean +/- standard deviation)"
    )
    cv_results = {}
    for name, classifier in models.items():
        pipeline = make_pipeline(classifier, preprocessor)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scorers,
            n_jobs=-1,
        )
        cv_results[name] = scores
        print(f"\n{name}")
        for metric in scorers:
            values = scores[f"test_{metric}"]
            print(f"  {metric.capitalize():<10} {values.mean():.3f} +/- {values.std():.3f}")

    # Compare only the three real algorithms when selecting by at-risk F1.
    best_name = max(
        (name for name in models if name != "Dummy Classifier"),
        key=lambda name: cv_results[name]["test_f1"].mean(),
    )
    best_model = make_pipeline(models[best_name], preprocessor)
    best_model.fit(X_train, y_train)

    # Use the test set only now, after model selection is complete.
    test_predictions = best_model.predict(X_test)
    print(f"\nSelected model: {best_name}")
    print("Final test results for the at-risk class")
    print(f"  Accuracy:  {accuracy_score(y_test, test_predictions):.3f}")
    print(
        f"  Precision: "
        f"{precision_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(f"  Recall:    {recall_score(y_test, test_predictions, zero_division=0):.3f}")
    print(f"  F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.3f}")
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                  Predicted Not At Risk  Predicted At Risk")
    print(f"Actual Not At Risk             {matrix[0, 0]:>3}                {matrix[0, 1]:>3}")
    print(f"Actual At Risk                 {matrix[1, 0]:>3}                {matrix[1, 1]:>3}")
    print(
        f"\nCorrectly identified {matrix[1, 1]} of "
        f"{matrix[1, 0] + matrix[1, 1]} at-risk students."
    )
    print(f"Missed {matrix[1, 0]} at-risk students.")


if __name__ == "__main__":
    main()
