# Predicting-Injury-Severity-using-Crash-Report-Drivers-Data
- Data 4380 Tabular Project
- **Group members**: Kelly Nguyen, Victoria Phan, Christopher Rodriguez
- **Dataset:** [Crash Reporting – Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)

## Overview
Motor vehicle crashes can result in injuries of varying severity. Understanding the factors associated with these outcomes can help identify patterns in crash data and provide insights into traffic safety.

This project analyzes crash report data from Montgomery County, Maryland, to investigate the relationship between crash characteristics, environmental conditions, vehicle characteristics, and injury severity.

Our goal is to identify factors associated with injury severity and evaluate whether these characteristics can help distinguish between different injury severity outcomes.

### Research Question

**What factors/characteristics contribute to injury severity from the years 2022-2025?**

### Project Objectives

- Clean and standardize inconsistent values in the crash report dataset.
- Explore the distribution of injury severity and its relationship with selected crash characteristics.
- Identify variables associated with differences in injury severity.
- Develop and evaluate classification models to examine how well injury severity can be distinguished using available crash and vehicle characteristics.
- Interpret the results while accounting for class imbalance, data limitations, and potential information leakage.

## Overview of approaches
The notebook follows a chronological modeling workflow:
1. **Load and inspect the data.** Read export.csv, inspect its structure, review missingness, and examine the raw records.
2. **Clean and harmonize fields.** Standardize column names and text, parse crash date/time, remove selected high-missingness columns, standardize injury-severity and categorical labels, group collision types, convert numeric fields, remove exact duplicates, and restrict records to 2022–2025.
3. **Align category labels across years.** The source changed several category labels in 2024. The notebook maps earlier labels to consistent categories so that models trained on earlier years can recognize corresponding values in later data.
4. **Create time features and prepare modeling data.** Derive crash hour, day of week, and month. Keep a cleaned dataset for descriptive analysis and a separate dataset with known injury severity for supervised modeling.
5. **Perform EDA.** Examine injury-severity distribution, missing values, collision types, weather, lighting, road-surface and traffic-control conditions, numeric variables, crash timing, and relationships among features.
6. **Establish a baseline.** Fit a logistic-regression baseline after imputation and preprocessing, then compare ways to address class imbalance.
7. **Compare models and imbalance strategies.** Experiments include logistic regression, random forest, histogram gradient boosting, and XGBoost, along with class weights, random over/undersampling, SMOTE, SMOTENC, and balanced random forest where appropriate.
8. **Validate chronologically.** The main development comparison trains on 2022–2023 and validates on 2024. The final evaluation uses 2025 as a later-year test set after retraining on 2022–2024.
9. **Select features and evaluate the final model.** Use validation performance and seed-to-seed variability to assess feature additions/removals, then report accuracy, balanced accuracy, macro F1, and class-level behavior.
The notebook’s documented final feature set contains 13 predictors: Collision Group, Weather, Light, Traffic Control, Vehicle Body Type, Vehicle Movement, Vehicle Going Dir, Crash Hour, Crash Day of Week, Crash Month, Latitude, Longitude, and Vehicle Age.

## Instructions for running the code
1. Start Jupyter or use Jupyter Notebook or Google Colab.
2. Open `(final)_Data_4380_notebook.ipynb`
3. Select Restart Kernel/Runtime and Run All (in Colab: Runtime → Restart session and run all).
4. Allow the notebook to finish. It performs cleaning, EDA, model comparisons, feature selection, and evaluation. Some model experiments may take several minutes depending on hardware.
**Important**: Run the notebook from the beginning in order. Later sections depend on variables created by earlier cleaning and preprocessing sections, so running isolated cells can lead to inconsistent results.

## Environment Setup and Dependencies
### Requirements

This project uses Python and Jupyter Notebook to perform data cleaning, exploratory data analysis (EDA), and machine learning for crash injury severity prediction.

**Prerequisites:**
- Python 3.10 or newer
- Jupyter Notebook
- The project dataset (`export.csv`)
- The dependencies listed in `requirements.txt`

### Main Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Load, clean, transform, and manipulate crash records |
| `numpy` | Numerical computations and array operations |
| `matplotlib` | Create data visualizations |
| `seaborn` | Generate statistical plots for exploratory data analysis |
| `missingno` | Visualize missing data patterns |
| `scikit-learn` | Build machine learning models, preprocess data, evaluate performance, and calculate feature importance |
| `imbalanced-learn` | Apply resampling techniques to address class imbalance, including SMOTE, random over/undersampling, and balanced random forests |
| `xgboost` | Train gradient-boosting classification models |
| `ipykernel` | Connect the Python environment to Jupyter kernels |
| `jupyter` | Run and interact with the notebook |

The versions of `scikit-learn` and `imbalanced-learn` are pinned to compatible versions to reduce potential dependency conflicts.

### Installation

1. Clone or download this repository and navigate to its directory.

   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. Create a virtual environment.

   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies.

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook.

   ```bash
   jupyter notebook
   ```

5. Open the project notebook and run the cells from beginning to end.

### Running the Project

Ensure that `export.csv` is available at the file path expected by the notebook. Start Jupyter Notebook, open the crash injury severity analysis notebook, and execute the cells in order.

The notebook performs data cleaning, exploratory data analysis, preprocessing, model development, model comparison, and evaluation. Some machine learning experiments may require additional processing time.

For reproducibility, use the provided `requirements.txt` file to install the specified dependencies before running the notebook.

