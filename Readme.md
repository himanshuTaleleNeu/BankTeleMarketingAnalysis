# Bank TeleMarketing Analysis

This README file provides a brief overview of the Python script for data analysis, preprocessing, and binary classification. The script covers essential steps for understanding and processing a dataset, including data exploration, cleaning, and feature engineering. Additionally, it demonstrates the creation and evaluation of binary classification models using machine learning algorithms.

## Python Environment Setup
Ensure you have a Python environment set up with the necessary libraries installed. You can create a Python environment using a tool like [conda](https://docs.conda.io/en/latest/), which allows you to manage dependencies efficiently. Here's an example of creating a conda environment:

```bash
conda create -n myenv python=3.7
conda activate myenv
```

You can install required packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pandas-profiling
```

## Data Preparation

1. Import the required Python libraries at the beginning of the script to perform data analysis and machine learning tasks.

2. Load the dataset into a Pandas DataFrame using `pd.read_csv()`. Ensure that you provide the correct file path.

3. Check the basic information about the dataset using `data.info()` and inspect the first few rows with `data.head()` to understand the data structure.

## Data Exploration (Descriptive Statistics)

1. Calculate basic summary statistics for numerical columns (mean, median, standard deviation) and examine the distribution of the 'age' and 'balance' variables.

2. Visualize the data distribution using histograms and bar charts.

## Data Cleaning

1. Identify and address missing values in the dataset. In this script, missing values in the 'age' and 'month' columns are handled by removing rows with missing 'age' values and filling missing 'month' values with the mode.

2. Clean ambiguous entries in columns like 'poutcome', 'job', and 'education' by replacing 'unknown' or 'other' values to ensure data clarity.

3. Convert unit values: Duration is converted from seconds to minutes, and the month column is converted from words to numbers.

4. Identify and handle outliers if necessary. In this script, we visualize and discuss potential outliers in the 'age' and 'balance' columns.

## Exploratory Data Analysis (EDA)

1. Visualize the distribution of the 'duration' and 'campaign' variables to gain insights into call durations and contact frequencies.

2. Explore the relationship between 'duration' and 'campaign' variables, highlighting the impact on client subscription.

3. Analyze the distribution of the target variable 'response' (subscription rate) using pie charts.

4. Visualize 'education' and 'marital' distributions.

5. Create scatter plots and heatmaps to visualize relationships between 'age,' 'balance,' and 'duration' variables.

6. Generate heatmaps to explore relationships between 'education,' 'marital' status, and 'poutcome' variables with the target variable 'response.'

## Feature Engineering

1. Select relevant columns for analysis and drop unwanted columns.

2. Encode categorical variables using one-hot encoding for 'job' and 'education' columns.

3. Map binary values 'yes' and 'no' to 1 and 0 for 'housing,' 'default,' and 'loan' columns.

4. Prepare the dataset for binary classification by encoding the target variable 'response' to 1 for 'yes' and 0 for 'no.'

## Binary Classification

1. Split the dataset into features (X) and the target variable (Y). Extract all columns except the last one for the feature matrix X and the last column for the target variable Y.

2. Train and evaluate binary classification models using scikit-learn:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Gaussian Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Random Forest

3. Evaluate model performance using metrics such as accuracy, F1 score, precision, recall, and ROC AUC score. Visualize confusion matrices to assess model performance further.

4. Provide an example of using a trained model to predict a new data point.

## Conclusion

This README outlines the steps and code for data analysis, preprocessing, and binary classification. The script is structured to guide you through each step, from data cleaning and exploration to model training and evaluation. The provided Python environment setup and library installation instructions help ensure smooth execution of the script.