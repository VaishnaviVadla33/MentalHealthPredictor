import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("D:/Sem4/ML/Project/Mental_Health_Dataset_processed.csv")

# Change dtypes of category
categorical_cols = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment',
                    'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
                    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview',
                    'care_options']

for col in categorical_cols:
    data[col] = data[col].astype('category')

####overall plot
# Create a 4x4 grid of subplots with smaller figure size for each plot
fig, axs = plt.subplots(4, 4, figsize=(10, 10))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Iterate over each categorical column and plot the distribution
for i, col in enumerate(data.select_dtypes(include='category').columns):
    sns.countplot(data=data, x=col, ax=axs[i], palette='Set3')
    axs[i].set_title(f'Distribution of {col}', fontsize=8)
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].tick_params(axis='x', labelsize=6, rotation=45)
    axs[i].tick_params(axis='y', labelsize=6)

# Adjust layout
plt.tight_layout()
plt.show()


# Distribution of categorical variables
for col in data.select_dtypes(include='category').columns:
    plt.figure(figsize=(8, 6))  # Reduced figure size
    sns.countplot(data=data, x=col, hue=col, order=data[col].value_counts().index, palette='Set3', legend=False) # Use 'Set3' palette
    plt.title(f'Distribution of {col}', fontsize=14)  # Reduced title font size
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col, fontsize=12)  # Reduced label font size
    plt.ylabel('Count', fontsize=12)
    plt.grid(False)  # Removed gridlines
    plt.tight_layout()
    plt.show()

# Export the cleaned data to a new CSV file
#data.to_csv("cleaned_data.csv", index=False)
