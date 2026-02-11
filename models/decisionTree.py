import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # for numbers -> fill missing with median
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # for categories -> fill missing + one hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # apply both transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # decision tree model
    model = DecisionTreeClassifier(
        max_depth=10,        # prevent it from growing too much
        random_state=67
    )

    # combine preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_decision_tree(X, y):

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=67,
        stratify=y
    )

    pipeline = build_pipeline(X)

    # train model
    pipeline.fit(X_train, y_train)

    # predict
    y_pred = pipeline.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", accuracy)
    print("F1 Score (Macro):", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline


# run everything
model = train_decision_tree(X, y)