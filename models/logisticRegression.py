import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

file_path = "studentData.csv" # The path to the csv dataset
df = pd.read_csv(file_path) # Load the dataset

df = df.dropna(subset=['Grades']) # Drop rows with missing grades
X = df.drop(columns=['Grades']) # Grab all columns except for the target (Grades)
y = df['Grades'] # Grab target

#############
# FUNCTIONS #
#############

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist() # Select numeric features
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist() # Select categorical features

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # Impute missing numeric values with the median
        ("scaler", StandardScaler()) # Scaling is good for logistic regression
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), # Impute missing categorical values with the median
        ("encoder", OneHotEncoder(handle_unknown="ignore")) # One-hot encode categorical values
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features), # Transform numeric features
            ("cat", categorical_transformer, categorical_features), # Transform categorical features
        ]
    )

    model = LogisticRegression(
        multi_class="multinomial", # Multiclass problem
        solver="lbfgs", # Generally good solver
        max_iter=1000, # Max number of iterations to solve for
        random_state=67
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=67 # 80/20 train-test split
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro") # Either macro or weighted works, but we don't actually care about what grades mean when weighed

    print("Accuracy:", accuracy)
    print("F1 Score (Macro):", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline

##################
# MAIN EXECUTION #
##################

model = train_logistic_regression(X, y)