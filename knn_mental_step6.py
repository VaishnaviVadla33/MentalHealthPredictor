import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

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

results = []
for num_features in range(1, 11):
    print(f"WITH {num_features} TOP FEATURES")

    # Select top features
    selector = SelectKBest(score_func=f_regression, k=num_features)
    X_selected = selector.fit_transform(X_train, y_train)

    for k in [3, 5, 7]:  # Number of neighbors for KNN
        print(f"WITH K={k}")

        for folds in [3, 5]:  # Number of folds for validation
            print(f"WITH {folds} FOLDS FOR VALIDATION")

            # Initialize KFold with k=3 or 5 folds
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            fold_number = 1

            for train_index, val_index in kf.split(X_selected):
                print(f"Fold {fold_number}")

                X_train_fold, X_val_fold = X_selected[train_index], X_selected[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                # Train a KNN regression model
                knn_reg = KNeighborsRegressor(n_neighbors=k)
                knn_reg.fit(X_train_fold, y_train_fold)

                # Make predictions on the validation set
                y_pred_fold = knn_reg.predict(X_val_fold)

                # Calculate Mean Squared Error (MSE), MAE, R2 for validation
                mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
                mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
                rsquared_fold = r2_score(y_val_fold, y_pred_fold)
                class_report_fold = classification_report(y_val_fold, np.round(y_pred_fold), output_dict=True)

                # Append metrics to results
                results.append({'dataset': 'training',
                                'num_features': num_features,
                                'k_neighbors': k,
                                'folds': folds,
                                'fold_number': fold_number,
                                'MSE': mse_fold,
                                'MAE': mae_fold,
                                'R2': rsquared_fold,
                                'Accuracy': class_report_fold['accuracy'],
                                'Precision': class_report_fold['weighted avg']['precision'],
                                'Recall': class_report_fold['weighted avg']['recall'],
                                'F1-Score': class_report_fold['weighted avg']['f1-score'],
                                'Specificity': class_report_fold['0']['recall'],
                                'BCA Score': 0.5*(class_report_fold['0']['recall'] + class_report_fold['1']['recall'])})
                fold_number += 1

            # Select top features for test set
            X_test_selected = selector.transform(X_test)

            # Make predictions on the test set
            y_pred_test = knn_reg.predict(X_test_selected)

            # Calculate Mean Squared Error (MSE) on the test set
            mse_test = mean_squared_error(y_test, y_pred_test)
            print("Mean Squared Error on testing data:", mse_test)

            # Calculate Mean Absolute Error (MAE) on the test set
            mae_test = mean_absolute_error(y_test, y_pred_test)
            print("Mean Absolute Error on testing data:", mae_test)

            # Calculate R-squared on the test set
            rsquared_test = r2_score(y_test, y_pred_test)
            print("R-squared on testing data:", rsquared_test)

            # Generate classification report for the test set
            class_report_test = classification_report(y_test, np.round(y_pred_test), output_dict=True)

            # Append metrics for test set to results
            results.append({'dataset': 'testing',
                            'num_features': num_features,
                            'k_neighbors': k,
                            'folds': folds,
                            'fold_number': 'testing data',
                            'MSE': mse_test,
                            'MAE': mae_test,
                            'R2': rsquared_test,
                            'Accuracy': class_report_test['accuracy'],
                            'Precision': class_report_test['weighted avg']['precision'],
                            'Recall': class_report_test['weighted avg']['recall'],
                            'F1-Score': class_report_test['weighted avg']['f1-score'],
                            'Specificity': class_report_test['0']['recall'],
                            'BCA Score': 0.5*(class_report_test['0']['recall'] + class_report_test['1']['recall'])})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define the metric to optimize
metric_to_optimize = 'Accuracy'

# Sort the results_df DataFrame based on the metric to optimize
sorted_results = results_df.sort_values(by=[metric_to_optimize], ascending=False)

# Select the top 5 best results
top_results = sorted_results.head(5)

# Display the top 5 best results
print("Top 5 Best Results:")
print(top_results)

# Save results to Excel
results_df.to_excel("D:/Sem4/ML/Project/Results/knn_model_results1_50ksamples.xlsx", index=False)
