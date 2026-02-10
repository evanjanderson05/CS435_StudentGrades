import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

file_path = "studentData.csv" # The path to the csv dataset
df = pd.read_csv(file_path) # Load the dataset

df = df.dropna(subset=['Grades']) # Drop rows with missing grades
X = df.drop(columns=['Grades']) # Grab all columns except for the target (Grades)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Grades'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y # 80/20 train-test split. Stratify to ensure same target distribution
)

#############
# FUNCTIONS #
#############


def majority_classifier(X, y):
    majority_class = pd.Series(y_train).mode()[0] # Find the most common target class

    y_pred = [majority_class] * len(y_test) # Only predict the most common target class

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro") # Either macro or weighted works, but we don't actually care about what grades mean when weighed

    print("Accuracy:", accuracy)
    print("F1 Score (Macro):", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0)) # zero-division = 0 to silence warnings

    return majority_class

##################
# MAIN EXECUTION #
##################

majority_class = majority_classifier(X, y)