# Bank TeleMarketing Analysis

## Introduction
This ReadMe file provides a detailed guide for using a Python script to analyze and build predictive models for a bank marketing campaign dataset. The dataset contains information about the marketing campaign conducted by a bank and the responses of clients to subscribe to a term deposit.

The key steps covered in this analysis include **Data Preprocessing, Exploratory Data Analysis (EDA), and Building Models** to predict whether a client will subscribe to a term deposit. The Python script utilizes various libraries, including pandas, numpy, scikit-learn, Pandas Profiling Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Random Forest, Decision Tree, and LightGBM. for data analysis and model building.

## Prerequisites
To run the script and reproduce the analysis, you will need the following:

Python environment with necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm).
A Jupyter Notebook or a code editor to execute the Python script.

## :spiral_notepad: Python Environment Setup
Ensure you have a Python environment set up with the necessary libraries installed. You can create a Python environment using a tool like [conda](https://docs.conda.io/en/latest/), which allows you to manage dependencies efficiently. Here's an example of creating a conda environment:

```bash
conda create -n myenv python=3.7
conda activate myenv
```

You can install required packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pandas-profiling
```

Instructions
Follow the step-by-step instructions below to use the provided Python script:

## :arrow_down: Step 1: Data Preprocessing
Import the necessary libraries by running the script.
Load the dataset by specifying the file path in the data = pd.read_csv() line. Make sure to replace the file path with your dataset's location.
Remove ambiguous values and missing data from the dataset, specifically in the 'age' and 'month' columns.
Handle outliers in the 'age' and 'balance' columns.
Convert the 'pdays' column values of -1 to NaN.
Remove rows with 'poutcome' containing 'other'.
Replace 'unknown' values in the 'job' and 'education' columns with 'other'.
Perform unit conversion on the 'duration' column to represent call duration in minutes.
Convert month names in the 'month' column to corresponding numbers.

## :chart_with_downwards_trend: Step 2: Exploratory Data Analysis (EDA)
Compute basic summary statistics for numerical columns ('age' and 'balance') and display the distribution of these variables.
Analyze and visualize the relationship between 'duration' and 'campaign'.
Visualize the distribution of the target variable 'response'.
Explore the distribution of 'education' and 'marital' categories.
Visualize the relationships between 'age', 'balance', and 'duration'.
Analyze subscription and contact rates by age group.
Analyze subscription rates by balance level.
Examine the response rate by job category.


## :anchor: Step 3: Data Preparation for Binary Classification
Create a clean dataset for binary classification, including only relevant columns and encoding target variable 'response' to binary values.
Split the dataset into features (X) and the target variable (Y).

## :snail: Step 4: Model Building and Evaluation
Train binary classification models using Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Random Forest, Decision Tree, and LightGBM.
Evaluate the model performance by calculating accuracy and generating confusion matrices.
Make predictions for a new data point using the trained models and display the response.

## :spider_web: Running the Script
To run the Python script, execute the code cells one by one in a Jupyter Notebook or a compatible code editor. Ensure that you have the necessary libraries installed in your Python environment.

## Conclusion

This script provides a comprehensive analysis of the bank marketing campaign dataset, including data preprocessing, EDA, and binary classification model building. By following the instructions in this ReadMe, you can explore and analyze the dataset, train different classification models, and make predictions based on new data.
