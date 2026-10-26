"""A simple Streamlit app for estimating a student's risk of failing math."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Find project files relative to this script, not the folder the app is run from.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "student-mat.csv"
IMPORTANCE_CHART = PROJECT_DIR / "images" / "feature_importance.png"

# Group inputs into sections that make sense to someone entering student details.
FEATURE_SECTIONS = {
    "Student background": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "Family and support": [
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "nursery",
        "higher",
        "internet",
    ],
    "Lifestyle and school life": [
        "traveltime",
        "studytime",
        "failures",
        "activities",
        "romantic",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
    ],
}

# Friendly descriptions make dataset abbreviations easier to understand.
FEATURE_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Home address type",
    "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing school",
    "guardian": "Student's guardian",
    "schoolsup": "Extra educational support",
    "famsup": "Family educational support",
    "paid": "Extra paid classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "traveltime": "Travel time to school",
    "studytime": "Weekly study time",
    "failures": "Past class failures",
    "famrel": "Family relationship quality",
    "freetime": "Free time after school",
    "goout": "Going out with friends",
    "Dalc": "Workday alcohol consumption",
    "Walc": "Weekend alcohol consumption",
    "health": "Current health",
    "absences": "School absences",
}

# Display friendly names while keeping each option's original dataset value.
OPTION_LABELS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"GT3": "More than 3 family members", "LE3": "3 or fewer family members"},
    "Pstatus": {"A": "Living apart", "T": "Living together"},
    "Mjob": {
        "at_home": "At home",
        "health": "Health care",
        "other": "Other",
        "services": "Services",
        "teacher": "Teaching",
    },
    "Fjob": {
        "at_home": "At home",
        "health": "Health care",
        "other": "Other",
        "services": "Services",
        "teacher": "Teaching",
    },
    "reason": {
        "course": "Course preference",
        "home": "Close to home",
        "other": "Other",
        "reputation": "School reputation",
    },
    "guardian": {"father": "Father", "mother": "Mother", "other": "Other"},
    "schoolsup": {"no": "No", "yes": "Yes"},
    "famsup": {"no": "No", "yes": "Yes"},
    "paid": {"no": "No", "yes": "Yes"},
    "activities": {"no": "No", "yes": "Yes"},
    "nursery": {"no": "No", "yes": "Yes"},
    "higher": {"no": "No", "yes": "Yes"},
    "internet": {"no": "No", "yes": "Yes"},
    "romantic": {"no": "No", "yes": "Yes"},
}

# Explain the source dataset's numeric scales beside the corresponding widgets.
FEATURE_HELP = {
    "Medu": "0 = none; 1 = primary (4th grade); 2 = 5th–9th grade; "
    "3 = secondary; 4 = higher education.",
    "Fedu": "0 = none; 1 = primary (4th grade); 2 = 5th–9th grade; "
    "3 = secondary; 4 = higher education.",
    "traveltime": "1 = under 15 min; 2 = 15–30 min; "
    "3 = 30 min–1 hour; 4 = over 1 hour.",
    "studytime": "Weekly study time: 1 = under 2 hours; 2 = 2–5 hours; "
    "3 = 5–10 hours; 4 = over 10 hours.",
    "failures": "Number of past class failures: 0 means none; "
    "1–3 indicate one, two, or three or more past failures.",
    "famrel": "Family relationship quality, rated from 1 (very bad) to 5 (excellent).",
    "freetime": "Free time after school, rated from 1 (very low) to 5 (very high).",
    "goout": "Going out with friends, rated from 1 (very low) to 5 (very high).",
    "Dalc": "Workday alcohol consumption, rated from 1 (very low) to 5 (very high).",
    "Walc": "Weekend alcohol consumption, rated from 1 (very low) to 5 (very high).",
    "health": "Current health status, rated from 1 (very bad) to 5 (very good).",
}


@st.cache_resource
def load_trained_model():
    """Load data and train once, using only the training portion of the split."""
    # The Mathematics file is a semicolon-separated CSV.
    data = pd.read_csv(DATA_FILE, sep=";")

    # Label students with a final grade below 10 as at risk.
    target = (data["G3"] < 10).astype(int)

    # Do not include any grades as model inputs.
    features = data.drop(columns=["G1", "G2", "G3"])

    # Create the same reproducible, stratified split used by train_model.py.
    X_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    # Identify text and numeric columns from the training features.
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns
    numerical_features = X_train.select_dtypes(include=["number"]).columns

    # Encode categories and scale numeric values inside the model pipeline.
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
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    # Return only training inputs so widgets use values available during fitting.
    return model, X_train


def make_input_form(X_train):
    """Collect one student's feature values using dataset-backed widgets."""
    student = {}
    with st.form("student_information"):
        st.write("Enter the student's information below.")

        # Dataset-backed options ensure categorical values match model training.
        for section, feature_names in FEATURE_SECTIONS.items():
            section_features = [
                name for name in feature_names if name in X_train.columns
            ]
            if not section_features:
                continue

            st.markdown(f"#### {section}")
            columns = st.columns(2)
            for index, feature in enumerate(section_features):
                values = X_train[feature].dropna()
                label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

                with columns[index % 2]:
                    if pd.api.types.is_numeric_dtype(X_train[feature]):
                        minimum = int(values.min())
                        maximum = int(values.max())
                        default = int(values.median())
                        student[feature] = st.slider(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            help=FEATURE_HELP.get(feature),
                            key=f"input_{feature}",
                        )
                    else:
                        # Sorting creates a stable, easy-to-scan dropdown list.
                        options = sorted(values.unique().tolist(), key=str)
                        student[feature] = st.selectbox(
                            label,
                            options=options,
                            format_func=lambda value, choices=OPTION_LABELS.get(
                                feature, {}
                            ): choices.get(value, str(value).replace("_", " ").title()),
                            help=FEATURE_HELP.get(feature),
                            key=f"input_{feature}",
                        )

        submitted = st.form_submit_button("Predict Student Risk", type="primary")
    return student, submitted


def main():
    st.set_page_config(
        page_title="Student Success Predictor",
        page_icon="🎓",
        layout="wide",
    )
    st.title("Student Success Predictor")
    st.write(
        "Estimate whether a student may be at risk of failing Mathematics "
        "using their background and school-life information."
    )
    st.info(
        "Predictions are estimates, not certainties. Do not use them as the "
        "sole basis for academic decisions."
    )

    # Streamlit caches this trained pipeline so form interactions do not retrain it.
    model, X_train = load_trained_model()
    student, submitted = make_input_form(X_train)

    if submitted:
        # Make a one-row DataFrame with the same named inputs used for training.
        student_data = pd.DataFrame([student], columns=X_train.columns)
        prediction = int(model.predict(student_data)[0])
        at_risk_probability = float(model.predict_proba(student_data)[0][1])

        st.subheader("Prediction")
        result_columns = st.columns([1, 2])
        with result_columns[0]:
            if prediction == 1:
                st.error("### At Risk\nThe model estimates this student may need support.")
            else:
                st.success("### Not At Risk\nThe model does not flag this student as at risk.")
        with result_columns[1]:
            st.metric(
                "Estimated probability of being at risk",
                f"{at_risk_probability:.1%}",
            )
            st.progress(at_risk_probability)

    # Show the existing feature-importance chart only when the user expands it.
    with st.expander("Explore feature importance"):
        if IMPORTANCE_CHART.exists():
            st.image(
                str(IMPORTANCE_CHART),
                caption=(
                    "Permutation importance shows predictive usefulness, "
                    "not causation."
                ),
                width="stretch",
            )
        else:
            st.info(
                "The feature importance chart is not available yet. "
                "Run feature_importance.py to create it."
            )


if __name__ == "__main__":
    main()
