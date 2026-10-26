"""Show which student features are useful for identifying at-risk students."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Find the dataset and output chart relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "student-mat.csv"
CHART_FILE = PROJECT_DIR / "images" / "feature_importance.png"


def main():
    # The dataset uses semicolons between values rather than commas.
    data = pd.read_csv(DATA_FILE, sep=";")

    # A final grade below 10 means the student is at risk of failing.
    target = (data["G3"] < 10).astype(int)

    # Remove all grades from the inputs so the model cannot use grade data.
    features = data.drop(columns=["G1", "G2", "G3"])

    # Use the same stratified split as the training script for a fair comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    # Separate text categories from numbers so each type gets suitable processing.
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns
    numerical_features = X_train.select_dtypes(include=["number"]).columns

    # One-hot encode categories and scale numeric values. Dense output is
    # compatible with HistGradientBoostingClassifier.
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

    # Keep preprocessing and the classifier together in a pipeline. Fitting
    # this pipeline on training data prevents test data from leaking into it.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)

    # Shuffle one original feature at a time on the held-out test set. The drop
    # in F1-score estimates how useful each feature is for identifying class 1.
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="f1",
    )

    # Sort features from most to least important and print the top ten.
    results = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_importance": importance.importances_mean,
            "std_importance": importance.importances_std,
        }
    ).sort_values("mean_importance", ascending=False)
    top_ten = results.head(10)

    print("Top 10 permutation importances (scoring = at-risk F1):")
    for row in top_ten.itertuples(index=False):
        print(
            f"{row.feature:<15} "
            f"mean = {row.mean_importance:.4f}, "
            f"std = {row.std_importance:.4f}"
        )
    print(
        "\nPermutation importance measures predictive usefulness, "
        "not causation."
    )

    # A horizontal chart is easier to read when feature names are long.
    chart_data = top_ten.sort_values("mean_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        chart_data["feature"],
        chart_data["mean_importance"],
        xerr=chart_data["std_importance"],
        capsize=4,
    )
    ax.set_xlabel("Mean decrease in F1-score after shuffling")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Feature Importances for At-Risk Prediction")
    fig.tight_layout()

    # Create the output folder if it does not already exist, then save the chart.
    CHART_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_FILE, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {CHART_FILE}")


if __name__ == "__main__":
    main()
