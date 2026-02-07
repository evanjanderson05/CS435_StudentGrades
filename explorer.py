import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = "studentData.csv" # The path to the csv dataset
df = pd.read_csv(file_path) # Load the dataset


#############
# FUNCTIONS #
#############

def featureDetails(col, graph=False): # Provides descriptive statistics for a given feature (col). Optional graph parameter defaulting to False
    print(f"\n{col}") # Print the feature name
    print(f"Type: {df[col].dtype}") # Print the type of the feature
    print(f"Missing: {df[col].isnull().sum()}") # Print the number of missing values

    if pd.api.types.is_numeric_dtype(df[col]): # Check for numeric features
        print(df[col].describe()) # Describe numeric features
        if (graph): # If graph parameter is True, graph the feature
            graphNumeric(col)
    else:
        print(df[col].value_counts(dropna=False).head(10)) # For categorical features, print the 10 most common values (including missing values)
        if (graph): # If graph parameter is True, graph the feature
            graphCategorical(col)

def graphNumeric(col):
    series = df[col].dropna() # Drop NA values

    min_val = int(series.min()) # Find min
    max_val = int(series.max()) # Find max
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1) # Produce edges such as 17.5-18.5 to capture values of 18; offers visual clarity

    plt.figure() # Create figure
    plt.hist(series, bins=bins, edgecolor="black", linewidth=2) # Plot a histogram using the created bins
    plt.xticks(range(min_val, max_val + 1)) # Manually set the tickmarks to fit the integer range of the feature
    plt.title(f"{col}: Histogram")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout() # Adjust spacing
    plt.show()

def graphCategorical(col):
    series = df[col].value_counts(dropna=False).head(10) # Grab the 10 most frequent values

    plt.figure() # Create figure
    series.plot(kind="bar", edgecolor="black", linewidth=2) # Create bar graph
    plt.title(f"{col}: Top 10 Categories")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right") # Rotate labels and align to the right for visual clarity
    plt.tight_layout() # Adjust spacing
    plt.show()

def corrMatrix(graph = False): # Optional graph parameter for ease of examination
    if not graph: # If graph is false, exit the function
        return
    df_encoded = df.copy() # Copy the df
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns: # Select categorical columns
        df_encoded[col] = df_encoded[col].astype("category").cat.codes # Convert columns into categorical type for heatmapping

    corr_matrix = df_encoded.corr() # Create correlation matrix

    plt.figure() # Create figure
    sns.heatmap(
        corr_matrix, # Use corr_matrix data
        annot=True, # Write numbers onto each cell
        fmt=".2f", # Two decimals
        cmap="coolwarm", # Blue for negative, red for positive, white for neutral
        vmin=-1, vmax=1 # Min correlation -1, max correlation 1 (for colors)
    )
    plt.title("Feature Correlation Matrix")
    plt.show()


##################
# MAIN EXECUTION #
##################

for col in df.columns:
    featureDetails(col, False) # Get the feature details for each feature

corrMatrix(True) # Graph the correlation matrix


###########
# ARCHIVE #
###########

# print(df.shape) # Print the shape of the dataset. For the student data, there are 10064 records and 35 features

# print(df.isnull().sum()) # Print the number of missing values for each feature

# print(df.describe()) # Describes numeric features, including mean/std/50%/max, etc.

# print(df.describe(include=["object", "category", "bool"]).T) # Describes categorical features using count/unique/top (mode)/freq (frequency of mode). Transposed for ease of reading