import pandas as pd # Import pandas for data manipulation
import numpy as np # Import numpy for numerical operations
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced visualizations
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import StandardScaler for feature scaling and OneHotEncoder for categorical encoding
from sklearn.impute import SimpleImputer # Import SimpleImputer for handling missing values
from sklearn.compose import ColumnTransformer # Import ColumnTransformer for applying different transformations to different columns
from sklearn.pipeline import Pipeline # Import Pipeline for chaining multiple processing steps

# 1. Load Titanic-Dataset.csv from the project folder
df = pd.read_csv('Dataset/Titanic-Dataset.csv') # Load the dataset into a pandas DataFrame

# 2. Show initial dataset information
print("Dataset Shape:") # Print header for dataset shape
print(df.shape) # Display the number of rows and columns
print("\nData Types:") # Print header for data types
print(df.info()) # Display column names and their data types
print("\nNumber of Missing Values:") # Print header for missing values
print(df.isnull().sum()) # Display the count of missing values per column
print("\nStatistical Summary:") # Print header for statistical summary
print(df.describe()) # Display descriptive statistics for numerical columns

# 3. Handle missing values
# Age: fill with median
df['Age'].fillna(df['Age'].median(), inplace=True) # Fill missing 'Age' values with the median age
# Cabin: drop
df.drop('Cabin', axis=1, inplace=True) # Remove the 'Cabin' column due to many missing values
# Embarked: fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) # Fill missing 'Embarked' values with the most frequent port

# Define categorical and numerical features for preprocessing
categorical_features = ['Sex', 'Embarked'] # List of columns to be one-hot encoded
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch'] # List of columns to be standardized

# Create a column transformer for preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Apply StandardScaler to numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Apply OneHotEncoder to categorical features
    ],
    remainder='passthrough' # Keep other columns not specified in transformers
)

# Create a pipeline for preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)]) # Create a pipeline with the preprocessor

# Apply preprocessing to the dataset
df_processed = pipeline.fit_transform(df) # Fit and transform the data using the defined pipeline

# Get feature names after one-hot encoding
ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features) # Get names for one-hot encoded columns
all_feature_names = numerical_features + list(ohe_feature_names) # Combine numerical and one-hot encoded feature names
# Add remaining columns that were passed through (e.g., 'PassengerId', 'Survived', 'Pclass', 'Name', 'Ticket')
# Identify columns that were not transformed
remaining_columns = [col for col in df.columns if col not in numerical_features + categorical_features] # Get names of columns not processed
all_feature_names += remaining_columns # Add remaining column names to the feature list

# Convert the processed array back to a DataFrame
df_cleaned = pd.DataFrame(df_processed, columns=all_feature_names) # Create a new DataFrame from processed data

# Ensure 'Survived' is in the cleaned DataFrame and is an integer type
if 'Survived' in df_cleaned.columns: # Check if 'Survived' column exists
    df_cleaned['Survived'] = df_cleaned['Survived'].astype(int) # Convert 'Survived' to integer type

# 6. Visualize data
# Boxplots for numerical columns to show outliers
plt.figure(figsize=(15, 10)) # Set the figure size for multiple plots
for i, col in enumerate(numerical_features): # Iterate through each numerical feature
    plt.subplot(2, 2, i + 1) # Create a subplot for each boxplot
    sns.boxplot(y=df_cleaned[col]) # Generate a boxplot for the current numerical column
    plt.title(f'Boxplot of {col}') # Set the title for the boxplot
plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.savefig('Output/boxplots_numerical_features.png') # Save the boxplots to the 'Output' folder
plt.show() # Display the boxplots

# Heatmap of correlation
plt.figure(figsize=(12, 10)) # Set the figure size for the heatmap
# Select only numeric columns for correlation calculation
numeric_cols_for_corr = df_cleaned.select_dtypes(include=np.number).columns.tolist() # Get all numeric columns for correlation
sns.heatmap(df_cleaned[numeric_cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f") # Generate a heatmap of feature correlations
plt.title('Correlation Heatmap of Features') # Set the title for the heatmap
plt.savefig('Output/correlation_heatmap.png') # Save the heatmap to the 'Output' folder
plt.show() # Display the heatmap

# Bar chart of Survived vs Sex
plt.figure(figsize=(8, 6)) # Set the figure size for the bar chart
# Create a bar plot of survival rate by sex (using Sex_male for clarity)
sns.barplot(x='Sex_male', y='Survived', data=df_cleaned, palette='viridis') # Generate bar plot for survival rate by sex
plt.title('Survival Rate by Sex (0 = Female, 1 = Male)') # Set the title for the bar chart
plt.xlabel('Sex (0 = Female, 1 = Male)') # Set the x-axis label to explain encoding
plt.ylabel('Survival Rate') # Set the y-axis label
plt.xticks(ticks=[0, 1], labels=['Female', 'Male']) # Set x-axis ticks and labels for clarity
plt.savefig('Output/survival_rate_by_sex.png') # Save the bar chart to the 'Output' folder
plt.show() # Display the bar chart

# 7. Export cleaned dataset as cleaned_titanic_dataset.csv
df_cleaned.to_csv('Output/cleaned_titanic_dataset.csv', index=False) # Save the processed DataFrame to a CSV file within the 'Output' folder
print("\nCleaned dataset exported as 'Output/cleaned_titanic_dataset.csv'") # Confirm export path
