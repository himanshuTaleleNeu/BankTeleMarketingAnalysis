# Bank TeleMarketing Analysis

## Introduction
This ReadMe file provides a detailed guide for using a Python script to analyze and build predictive models for a bank marketing campaign dataset. The dataset contains information about the marketing campaign conducted by a bank and the responses of clients to subscribe to a term deposit.

## Project Motivation
Term deposits serve as a key revenue stream for banks, representing cash investments with fixed interest rates over a specified term. To promote these deposits, banks employ diverse outreach strategies, including telephonic marketing. Despite its effectiveness, telephonic campaigns require substantial resources. This project aims to predict client subscriptions to term deposits, enhancing the efficiency of telephonic marketing by targeting potential subscribers more strategically.

Using data from phone-based marketing campaigns by a Portuguese bank, the project focuses on building a classification model. This model predicts whether a client will subscribe to a term deposit, empowering the bank to optimize telephonic efforts by prioritizing individuals with a higher likelihood of engagement.

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

## :arrow_down:  Step 1: Data Preprocessing
Import the necessary libraries by running the script.
Load the dataset by specifying the file path in the data = pd.read_csv() line. Make sure to replace the file path with your dataset's location.
Remove ambiguous values and missing data from the dataset, specifically in the 'age' and 'month' columns.
Handle outliers in the 'age' and 'balance' columns.
Convert the 'pdays' column values of -1 to NaN.
Remove rows with 'poutcome' containing 'other'.
Replace 'unknown' values in the 'job' and 'education' columns with 'other'.
Perform unit conversion on the 'duration' column to represent call duration in minutes.
Convert month names in the 'month' column to corresponding numbers.

## :chart_with_downwards_trend:  Step 2: Exploratory Data Analysis (EDA)
Compute basic summary statistics for numerical columns ('age' and 'balance') and display the distribution of these variables.
Analyze and visualize the relationship between 'duration' and 'campaign'.
Visualize the distribution of the target variable 'response'.
Explore the distribution of 'education' and 'marital' categories.
Visualize the relationships between 'age', 'balance', and 'duration'.
Analyze subscription and contact rates by age group.
Analyze subscription rates by balance level.
Examine the response rate by job category.


## :anchor:  Step 3: Data Preparation for Binary Classification
Create a clean dataset for binary classification, including only relevant columns and encoding target variable 'response' to binary values.
Split the dataset into features (X) and the target variable (Y).

## :snail:  Step 4: Model Building and Evaluation
Train binary classification models using Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Random Forest, Decision Tree, and LightGBM.
Evaluate the model performance by calculating accuracy and generating confusion matrices.
Make predictions for a new data point using the trained models and display the response.

## :spider_web:  Running the Script
To run the Python script, execute the code cells one by one in a Jupyter Notebook or a compatible code editor or if you are using Visual code then by using Terminal you can Navigate to the project directory followed with filename.py. Ensure that you have the necessary libraries installed in your Python environment.

## :computer: Streamlit Installation Guide
Streamlit is a powerful Python library for creating interactive web applications with minimal code. Follow these steps to install Streamlit and run the Bank TeleMarketing Analysis script as a web application:

##### Step 1: Install Streamlit
Open a terminal and run the following command to install Streamlit:

```bash
pip install streamlit
```
##### Step 2: Run the Streamlit App Locally
Once Streamlit is installed, you can run the Streamlit app locally. Navigate to the project directory and run the following command:

```bash
streamlit run your_streamlit_script.py
```
##### Step 3: Access the Streamlit App in the Browser
After running the above command, Streamlit will provide a local URL (usually http://localhost:8501). Open this URL in your web browser to access the Streamlit app and interact with the Bank TeleMarketing Analysis in a web-based interface.

## :rocket: Heroku Deployment Process

##### Step 1: Install Heroku CLI
Make sure you have the Heroku Command Line Interface (CLI) installed or you can connect it through Github on Heruko Website. You can download it from the official Heroku website.

Follow these steps to deploy the Bank TeleMarketing Analysis project on Heroku:

##### Step 2: Login to Heroku
Open a terminal and log in to your Heroku account using the following command:

```bash
heroku login
```
Follow the prompts to enter your Heroku credentials.

##### Step 3: Initialize a Git Repository 
If this project is not already a Git repository, initialize one using the following commands:

```bash
git init
git add .
git commit -m "BankTeleMarketing initial Commit"
```

##### Step 4: Create a Heroku App
Run the following command to create a new Heroku app:

```bash
heroku create banktelemarketing
```

Replace banktelemarketing with a unique name for your Heroku app. Heroku will provide you with a URL for your deployed app (e.g., https://banktelemarketing-8d9b5661ea18.herokuapp.com/)


##### Step 5: Create a Procfile
Ensure that you have a Procfile in your project root directory. The Procfile specifies the commands that are executed by the app on startup. For a Python project, the Procfile might look like this:

```bash
web: sh setup.sh && streamlit run Streamlit/BankTeleMarketingUI.py
```

###### Here please replace BankTeleMarketingUI.py with the actual name of your main Streamlit UI or python script.

##### Step 6: Create a requirements.txt file
If you haven't already, generate a requirements.txt file that lists all the dependencies of your project. You can use the following command:

```bash
pip freeze > requirements.txt
```
You can refer the above requirements.txt

##### Step 7: Commit Changes
Commit the changes to your Git repository:

```bash
git add .
git commit -m "Heroku deployment setup"
```

##### Step 8: Push to Heroku
Push your code to the Heroku remote repository:

```bash
git push heroku master/main
```

##### Step 9: Open the App
After the deployment is successful, open your app in the browser using:

```bash
heroku open
```

## Conclusion

This script provides a comprehensive analysis of the bank marketing campaign dataset, encompassing crucial steps such as data preprocessing, exploratory data analysis (EDA), and binary classification model building. By following the instructions in this ReadMe, you can thoroughly explore and analyze the dataset, train diverse classification models, and make predictions based on new data.

Additionally, the project has been successfully deployed on Heroku, allowing users to access and interact with the analysis through a web interface. The deployment process involves setting up the Heroku environment, creating a Heroku app, and pushing the project code to the Heroku remote repository. Whether you are a contributor looking to enhance the project or a user interested in exploring the analysis, the Heroku deployment ensures easy access and usage. The deployed application can be accessed through the provided Heroku app URL.

Feel free to fork the repository, contribute to the project, or simply deploy it for your use. Happy exploring and modeling!
