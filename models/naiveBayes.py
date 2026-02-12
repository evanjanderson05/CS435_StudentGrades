import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# get folder where this file is
HERE = os.path.dirname(os.path.abspath(__file__))

# go back one level to get the csv file
file_path = os.path.join(HERE, "..", "studentData.csv")

# load dataset
df = pd.read_csv(file_path)

# remove rows where Grades is missing
df = df.dropna(subset=["Grades"])

# split into features and target
X = df.drop(columns=["Grades"])
y = df["Grades"]


def build_pipeline(X):

    # separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    # MultinomialNB requires non-negative inputs
    def clip_nonnegative(a):
        return np.clip(a, 0, None)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(clip_nonnegative))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = MultinomialNB(alpha=1.0)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_naive_bayes(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=67,
        stratify=y
    )

    pipeline = build_pipeline(X)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", accuracy)
    print("F1 Score (Macro):", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline


# run everything
model = train_naive_bayes(X, y)
