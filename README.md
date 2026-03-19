# Mental Health Data Pipeline

A comprehensive data engineering pipeline for processing, transforming, and analyzing mental health survey data. This project implements a complete data lifecycle workflow from raw data ingestion through quality assurance and predictive analytics.

## Project Overview

This pipeline processes mental health assessment data to extract meaningful insights and patterns. The system follows a structured approach to handle data quality issues, perform exploratory analysis, and generate predictive models for mental health indicators.

**Type:** Data Engineering & Analytics Pipeline  
**Dataset:** Mental Health Assessment Survey  
**Processing Stages:** 8 sequential transformation steps  
**Output:** Structured processed data + Analytical insights

---

## Data Pipeline Architecture

The project implements a staged data processing pipeline:

```
Raw Data → Cleaning & Preprocessing → EDA & Visualization → 
Predictive Modeling → Results & Insights
```

### Pipeline Stages

| Step | Module | Purpose |
|------|--------|---------|
| **1** | `clean_mental_step1.py` | Data ingestion, type conversion, missing value handling |
| **2** | `visualization_eda_step2.py` | Exploratory analysis and distribution visualization |
| **3** | `logistic_mental_step3.py` | Logistic regression analysis |
| **4** | `decision_mental_step4.py` | Decision tree classification |
| **5** | `random_mental_step5.py` | Random forest ensemble approach |
| **6** | `knn_mental_step6.py` | K-nearest neighbors classification |
| **7** | `support_mental_step7.py` | Support vector machine analysis |
| **8** | `ensemble_method_mental_step8.py` | Multi-model ensemble predictions |

---

## Data Processing

### Input Data
- **Source File:** `Mental_Health_Dataset.csv`
- **Format:** Tabular (CSV)
- **Key Attributes:** Demographic, behavioral, and health-related indicators

### Data Cleaning & Transformation

**Step 1 - Data Preparation:**
- Load raw data and validate schema
- Type conversion (categorical features encoding)
- Missing value imputation:
  - Categorical fields: Mode-based imputation
  - Numeric fields: Mean-based imputation
- Remove non-analytical columns (e.g., Timestamp)

**Categorical Features Processed:**
Gender, Country, Occupation, Employment Status, Family History, Treatment Status, Indoor Duration, Stress Levels, Habit Changes, Health History, Mood Patterns, Coping Mechanisms, Work Interest, Social Interaction, Interview Completion, Care Options

**Output:** `Mental_Health_Dataset_processed.csv`

### Exploratory Data Analysis (EDA)

**Step 2 - Data Exploration:**
- Distribution analysis across all categorical dimensions
- Frequency counts and pattern identification
- Statistical summaries for data understanding
- Visualization outputs for stakeholder communication

---

## Analytical Models

The pipeline includes multiple classification approaches to understand mental health patterns:

- **Logistic Regression**: Linear relationship modeling for binary/multiclass outcomes
- **Decision Trees**: Rule-based decision paths for interpretability
- **Random Forest**: Ensemble aggregation for improved robustness
- **k-Nearest Neighbors**: Instance-based similarity matching
- **Support Vector Machines**: Optimal hyperplane classification
- **Ensemble Methods**: Hybrid model aggregation for enhanced predictions

### Model Outputs
Results stored in `/Results_after_all_clasifiers_executed/`:
- Model performance metrics
- Accuracy, precision, recall measurements
- Cross-validation results
- Comparative analysis across methods

---

## Project Structure

```
Mental_Health_Data_Pipeline/
├── Raw Data
│   ├── Mental_Health_Dataset.csv          # Original source data
│   └── Mental_Health_Dataset_processed.csv # Cleaned & transformed
├── Processing Scripts
│   ├── clean_mental_step1.py              # Data ingestion & cleaning
│   ├── visualization_eda_step2.py         # Analysis & visualization
│   ├── logistic_mental_step3.py           # Statistical modeling
│   ├── decision_mental_step4.py           # Tree-based analysis
│   ├── random_mental_step5.py             # Ensemble approach 1
│   ├── knn_mental_step6.py                # Instance-based modeling
│   ├── support_mental_step7.py            # SVM classification
│   └── ensemble_method_mental_step8.py    # Combined predictions
└── Results Artifacts
    └── Results_after_all_clasifiers_executed/
        ├── logistic_regression_results.xlsx
        ├── decision_tree_results1.xlsx
        ├── random_forest_results.xlsx
        ├── knn_model_results1.xlsx
        ├── svm_results.xlsx
        └── ensemble_results.xlsx
```

---

## Setup & Execution

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

### Installation

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Running the Pipeline

Execute pipeline stages sequentially:

```bash
# Stage 1: Data cleaning & preprocessing
python clean_mental_step1.py

# Stage 2: Exploratory analysis
python visualization_eda_step2.py

# Stage 3-8: Individual model analyses
python logistic_mental_step3.py
python decision_mental_step4.py
python random_mental_step5.py
python knn_mental_step6.py
python support_mental_step7.py
python ensemble_method_mental_step8.py
```

---

## Data Quality Measures

- **Missing Value Handling**: Strategic imputation based on data type
- **Type Validation**: Explicit conversion to appropriate data types
- **Schema Verification**: Consistency checks across transformation stages
- **Categorical Encoding**: Proper encoding of categorical dimensions
- **Output Validation**: Sample verification of processed datasets

---

## Key Insights & Outputs

The pipeline generates:

1. **Processed Dataset**: Clean, validated, analysis-ready data
2. **Exploratory Reports**: Visual distributions and frequency patterns
3. **Predictive Models**: Trained classifiers for pattern recognition
4. **Performance Metrics**: Comparative model evaluation
5. **Ensemble Predictions**: Aggregated results from multiple approaches

---

## Notes

- All file paths are configured for the project directory structure
- Models are executed independently; ensemble method combines insights from all approaches
- Results are exported as Excel workbooks for easy sharing and stakeholder review
- EDA visualizations help identify data patterns before modeling stages

---

📄 [View detailed documentation (PDF)](./Mental_Health_Analysis01.pdf)
