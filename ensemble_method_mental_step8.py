from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import KNeighborsClassifier


# Load the data
data = pd.read_csv("D:/Sem4/ML/Project/Mental_Health_Dataset_processed.csv")

# Reduce the dataset to 100,000 samples
data = data.sample(n=50000, random_state=42)

# Change dtypes of category
categorical_cols = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment',
                    'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
                    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview',
                    'care_options']

for col in categorical_cols:
    data[col] = data[col].astype('category')

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, dtype=int)

# Define features and target
X = data.drop('treatment_No', axis=1)  # Drop one of the treatment columns (e.g., 'treatment_No')
y = data['treatment_Yes']  # Use 'treatment_Yes' as the target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers for the ensemble
knn_classifier = KNeighborsClassifier()
logistic_classifier = LogisticRegression()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()

# Create the ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('knn', knn_classifier),
    ('logistic', logistic_classifier),
    ('dt', dt_classifier),
    ('rf', rf_classifier)
], voting='hard')

# Initialize the results list
results = []

for num_features in range(1, 11):
    print(f"WITH {num_features} TOP FEATURES")
    selector = SelectKBest(score_func=f_regression, k=num_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    for folds in [3, 5]:
        print(f"WITH {folds} FOLDS FOR VALIDATION")
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        fold_number = 1

        for train_index, val_index in kf.split(X_train_selected):
            print(f"Fold {fold_number}")
            X_train_fold, X_val_fold = X_train_selected[train_index], X_train_selected[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            ensemble_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = ensemble_model.predict(X_val_fold)

            # Calculate Mean Squared Error and Mean Absolute Error for validation set
            mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
            mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)

            # Generate classification report for validation set
            class_report_fold = classification_report(y_val_fold, y_pred_fold, output_dict=True)

            # Append metrics to results
            results.append({'dataset': 'training',
                            'num_features': num_features,
                            'folds': folds,
                            'fold_number': fold_number,
                            'Accuracy': class_report_fold['accuracy'],
                            'Precision': class_report_fold['weighted avg']['precision'],
                            'Recall': class_report_fold['weighted avg']['recall'],
                            'F1-Score': class_report_fold['weighted avg']['f1-score'],
                            'MSE': mse_fold,
                            'MAE': mae_fold})
            fold_number += 1

    ensemble_model.fit(X_train_selected, y_train)
    y_pred_test = ensemble_model.predict(X_test_selected)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    class_report_test = classification_report(y_test, y_pred_test, output_dict=True)

    results.append({'dataset': 'testing',
                    'num_features': num_features,
                    'folds': folds,
                    'fold_number': 'testing data',
                    'Accuracy': class_report_test['accuracy'],
                    'Precision': class_report_test['weighted avg']['precision'],
                    'Recall': class_report_test['weighted avg']['recall'],
                    'F1-Score': class_report_test['weighted avg']['f1-score'],
                    'MSE': mse_test,
                    'MAE': mae_test})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort results by Accuracy in descending order
sorted_results_ensemble = results_df[results_df['dataset'] == 'testing'].sort_values(by='Accuracy', ascending=False)

# Select the top 5 combinations of features and folds for the ensemble model
top_results_ensemble = sorted_results_ensemble.groupby(['num_features', 'folds']).head(1).head(5)

# Display the top 5 combinations for the ensemble model
print("Top 5 Combinations of Features and Folds for Ensemble Model:")
print(top_results_ensemble[['num_features', 'folds', 'Accuracy']])


# Save results to Excel
results_df.to_excel("D:/Sem4/ML/Project/Results/ensemble_results.xlsx", index=False)
