import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Data
data = pd.read_csv("D:/Sem4/ML/Project/Mental_Health_Dataset.csv")
print(data.info())

# Change dtypes of category
categorical_cols = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment',
                    'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
                    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview',
                    'care_options']

for col in categorical_cols:
    data[col] = data[col].astype('category')

data['Timestamp'] = data['Timestamp'].astype('category')


# Convert any necessary columns to appropriate numeric types
# For example, if 'Days_Indoors' contains strings like '1-14 days', you may want to convert it to a numeric value

# Print unique values of features that have missing percentage for either imputation or dropping of values
for col in data.columns:
    unique_vals = data[col].unique()
    print(f"Unique values of {col} are: {unique_vals}")

# Calculating missing value percentages
missing_percentage = data.isnull().mean() * 100
print("Percentage of missing values in each column:")
print(missing_percentage)

# Replace category data with median values and int/float with mean values
data = data.apply(lambda x: x.fillna(x.value_counts().index[0]) if x.dtype == "category" else x.fillna(x.mean()))

# Drop unnecessary columns if needed
data = data.drop(columns=['Timestamp'])

# Check info after preprocessing
print(data.info())

# Perform EDA (similar to what you did for the books dataset)
# Note: EDA will depend on the nature of the features in your mental health dataset

# Export the processed data to a new CSV file
data.to_csv("D:/Sem4/ML/Project/Mental_Health_Dataset_processed.csv", index=False)
