import pandas as pd

file_path = "studentData.csv" # The path to the csv dataset
df = pd.read_csv(file_path) # Load the dataset


#############
# FUNCTIONS #
#############

def featureDetails(col): 
    print(f"\n{col}") # Print the feature name
    print(f"Type: {df[col].dtype}") # Print the type of the feature
    print(f"Missing: {df[col].isnull().sum()}") # Print the number of missing values

    if pd.api.types.is_numeric_dtype(df[col]): # Check for numeric features
        print(df[col].describe()) # Describe numeric features
    else:
        print(df[col].value_counts(dropna=False).head(10)) # For categorical features, print the 10 most common values (including missing values)


##################
# MAIN EXECUTION #
##################

for col in df.columns:
    featureDetails(col) # Get the feature details for each feature


###########
# ARCHIVE #
###########

# print(df.shape) # Print the shape of the dataset. For the student data, there are 10064 records and 35 features

# print(df.isnull().sum()) # Print the number of missing values for each feature

# print(df.describe()) # Describes numeric features, including mean/std/50%/max, etc.

# print(df.describe(include=["object", "category", "bool"]).T) # Describes categorical features using count/unique/top (mode)/freq (frequency of mode). Transposed for ease of reading