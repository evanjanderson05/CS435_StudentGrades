import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

file_path = "studentData.csv" # The path to the csv dataset
df = pd.read_csv(file_path) # Load the dataset

df = df.dropna(subset=['Grades']) # Drop rows with missing grades
X = df.drop(columns=['Grades']) # Grab all columns except for the target (Grades)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Grades'])

#############
# FUNCTIONS #
#############

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist() # Select numeric features
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist() # Select categorical features

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")) # Impute missing numeric values with the median
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # Impute missing categorical values with the median
            ("encoder", OneHotEncoder(handle_unknown="ignore")) # One-hot encode categorical values
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features), # Transform numeric features
            ("cat", categorical_transformer, categorical_features), # Transform categorical features
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators = 100, # 100 trees
        learning_rate = 0.05, # low learning rate
        min_samples_leaf = 5, # leaf nodes must have at least 5 samples
        min_samples_split = 5,# node with less than 5 samples can't split inner node
        random_state = 67
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_gradient_boost(X, y):
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

model = train_gradient_boost(X, y)