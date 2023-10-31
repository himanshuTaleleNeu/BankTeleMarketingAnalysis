#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.executable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


# importing the dataset
data = pd.read_csv('/Users/himanshutalele/Desktop/Self/Bank_Campaign_Data/Bank_Marketing.csv', delimiter=';')
display(data)


# In[3]:


df = pd.DataFrame(data)
data.info()


# ### Step 1: Descriptive Statistics
# 
# Now we will ompute basic summary statistics for numerical variables, such as mean, median, and standard deviation and will examine the distribution of age and balance to understand the data.

# In[4]:


#Compute basic summary statistics
numerical_columns = ['age', 'balance']
categorical_columns = ['job']

# summary statistics for numerical columns
summary_statistics = df[numerical_columns].describe()

# summary statistics for the job column
job_summary = df['job'].value_counts()

# Print the summary statistics
print("Summary Statistics for Numerical Columns:")
print(summary_statistics)
print("\nSummary Statistics for 'job' Column:")
print(job_summary)

# Visualize data distribution
plt.figure(figsize=(15, 6))

# Histogram for age
plt.subplot(1, 3, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for balance
plt.subplot(1, 3, 2)
plt.hist(df['balance'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')

# Bar chart for 'job' column
plt.subplot(1, 3, 3)
job_summary.plot(kind='bar', color='coral')
plt.title('Job Distribution')
plt.xlabel('Job')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# ### Basic Summary Statistics
# 
# - **Age**: The age of clients in the dataset ranges from 18 to 95, with an average age of approximately 41 years. The majority of clients fall between the ages of 33 and 48, reflecting a relatively broad age distribution.
#  
# - **Balance**: The balance variable represents the account balance of clients, with a significant standard deviation indicating considerable variation. While the average balance is approximately 1362 units, the minimum and maximum account balances are -8019 and 102127 units, respectively.
#  
# - **Job**: The 'job' column categorizes clients into various job types. The most common job categories include 'blue-collar,' 'management,' and 'technician,' with 'blue-collar' being the largest group. There are also 'unknown' job categories in the dataset.
# 

# ### Step 2: Data Cleaning:
# 
# Here we will handle missing data by imputing or removing missing values, and will address any outliers that may affect the analysis.

# In[5]:


data.info()


# While there are missing values in age and month column, and also it contains entries such as 'unknown' and 'others' in column like poutcome, job, education which are used for analytical purposes, are treated as ambiguous values. So for these ambiguous entries we will remove from the dataset to ensure data clarity and reliability for analysis.

# ### 1. Age 

# In[6]:


#count the missing values in age column.
df.age.isnull().sum()


# In[7]:


#pring the shape of dataframe inp0
df.shape


# In[8]:


#calculate the percentage of missing values in age column.
float(100.0*20/45211)


# In[9]:


#drop the records with age missing in df and copy in new_df dataframe.
new_df = df[-df.age.isnull()].copy()
new_df


# Now as we have saved the results in new_df, hence forth we will perform new operation considering new_df

# ### 2. Month 

# In[10]:


#count the missing values in month column in new_df.
new_df.month.isnull().sum()


# In[11]:


#print the percentage of each month in the data frame new_df.
new_df.month.value_counts(normalize=True)


# In[12]:


#find the mode of month in new_df
month_mode = new_df.month.mode()[0]
month_mode


# In[13]:


# fill the missing values with mode value of month in new_df.
new_df.month.fillna(month_mode, inplace=True)
new_df.month.value_counts(normalize=True)


# In[14]:


#let's see the null values in the month column.
new_df.month.isnull().sum()


# ### 3.  y
#  first we will rename the y to response for better understanding

# In[15]:


nw_df = new_df.copy()

# Rename the 'y' column to 'response'
nw_df = nw_df.rename(columns={'y': 'response'})


# - So, we have removed missing values from column Age and Month. Also, in the next step missing values will be reevaluated. 

# In[16]:


#calculate the missing values in each column of data frame: nw_df.
nw_df.isnull().sum()


# - As we can see after importing the dataset, in pdays column there are -1 values, so we will replace those -1 with NaN

# In[17]:


nw_df.pdays.describe()


# In[18]:


nw_df['pdays'] = nw_df['pdays'].mask(nw_df['pdays'] < 0, pd.NA)
nw_df.pdays.describe()


# - Last step : to remove the 'other' from poutcome, job, education 

# In[19]:


# Here we will remove the rows which column 'poutcome' contains 'other'
nw_df = nw_df[nw_df['poutcome'] != 'other']


# In[20]:


# Here we will copy the data in new data frame and will replace 'unknown' with 'other' in the 'job' and 'education' columns
new_df1 = nw_df.copy()

new_df1['job'] = new_df1['job'].replace('unknown', 'other')
new_df1['education'] = new_df1['education'].replace('unknown', 'other')

# New DF results
new_df1


# In next step we will anaylse and drop the unwanted column, unit conversion for standardising the values
# 
# 1. Will remove the contact column
# 2. will change unit conversion in duration column from seconds to min 
# 3. will change the month column from words to number.

# In[21]:


# 1. will drop the conact column and will save in new dataframe i.e n_df1
n_df1 = new_df1.drop('contact', axis=1)


# In[22]:


#2. Unit conversion in duration column from seconds to min 
n_df1['duration'] = n_df1['duration'].apply(lambda n:n/60).round(2)


# In[23]:


# 3. Month column from words to number
# Here we will first create  a dictionary to map month names to their corresponding integers
mth  = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

n_df1["month"] = n_df1["month"].map(mth)


# ## Checking Outliers
#  - As a next step of Data Cleaning we will check if there are any outliers.
#  
# # 1. Age

# In[24]:


n_df1.age.describe()


# # Histogram

# In[25]:


n_df1.age.plot.hist()
plt.show()


# # Boxplot 

# In[26]:


sns.boxplot(n_df1.age)
plt.show()


# # 2. Balance

# In[27]:


n_df1.balance.describe()


# In[28]:


plt.figure(figsize=[8,2])
sns.boxplot(n_df1.balance)
plt.show()


# Now we will use Quantiles to divide a dataset into equal parts, ehich will provide  us the insights into the distribution of data. 

# In[29]:


n_df1.balance.quantile([0.5,0.7,0.9,0.95,0.99])


# In[30]:


profile = ProfileReport(n_df1, title='Data Profiling Report')


# In[31]:


profile


# ### Step 3. Exploratory Data Analysis

# # 1. We will visualize the distribution of 'duration' & 'campaign

# In[32]:


duration_distance_plot = n_df1[['duration','campaign']].plot(kind = 'box', color= 'red',
                                                      figsize = (8,8),
                                                      subplots = True, layout = (1,2),
                                                      sharex = False, sharey = False,
                                                      title='The Distribution of Duration and Campaign')
plt.show()


# - **Duration**: So, here we can see from the above box plot that most calls are short, with a median duration of 3 minutes. However, there are outliers lasting 10 to 40 minutes, which require further investigation.
# 
# - **Campagin**: While many clients were contacted once or twice, a few had up to 58 contacts, indicating special circumstances.

# # 2. Relationship between Duration & Campaign

# In[33]:


# Create a scatter plot with Seaborn
duration = sns.lmplot(x='duration', y='campaign', data=n_df1,
                     hue='response',
                     fit_reg=False,  # Disable regression line
                     scatter_kws={'alpha': 0.8}, height=7)

# axis limits for better visualization
plt.axis([0, 65, 0, 65])

plt.ylabel('Number of Calls')
plt.xlabel('Duration of Calls')

plt.title('Number and Duration of Calls')

# horizontal dashed line at y=5
plt.axhline(y=5, linewidth=2, color="brown", linestyle='--')

# Annotate the plot to highlight a point
plt.annotate('Higher subscription rate when calls <5', xytext=(35, 13),
             arrowprops=dict(color='blue', width=1), xy=(30, 6))

# Show the plot
plt.show()


# - In the scatter plot, "yes" clients subscribed to term deposits, while "no" clients did not. The plot shows two distinct clusters: "yes" clients were contacted fewer times and had longer call durations compared to "no" clients. Notably, after five campaign calls, clients are more likely to decline unless the call duration is extended. Most "yes" clients were contacted fewer than 10 times. 
# 
# - This implies that the bank should avoid calling a client more than five times, as excessive calls may lead to dissatisfaction.

# # 3. Response of target variable

# In[34]:


n_df1.response.value_counts(normalize=True)


# In[35]:


# Count the values and create a pie chart with percentage labels
n_df1['response'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.title('Response Distribution')

plt.show()


# - Here we can see that response distribution of Yes is 11.5% and No is 88.5%

# # 4. Education 

# In[36]:


# values and a pie chart with percentage labels
n_df1['education'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.title('Education Distribution')
plt.show()


# # 5. Visualization between age, balance and duration

# In[37]:


# pairplot with numerical values on the diagonal
pairplot = sns.pairplot(data=n_df1, vars=["balance", "age", "duration"])

for ax in pairplot.axes.flat:
    ax.annotate(ax.get_title(), xy=(0.5, 1.02), xycoords='axes fraction', fontsize=12, ha='center')

plt.show()


# - Heatmap for better visualization

# In[38]:


correlation_matrix = n_df1[["balance", "age", "duration"]].corr()

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="BrBG", fmt=".2f", linewidths=0.5)

plt.title('Correlation Heatmap')

plt.show()


# # 6. Marital vs response rate

# In[39]:


# Convert the binary 'response' column to a numerical format (0 for 'no', 1 for 'yes')
n_df1['response_numeric'] = n_df1['response'].map({'no': 0, 'yes': 1})

marital_response_mean = n_df1.groupby("marital")["response_numeric"].mean()

marital_response_mean.plot(kind='barh')
plt.xlabel('Response')
plt.ylabel('Marital Status')
plt.title(' Response vs Marital Status')
plt.show()


# # 7. Education vs poutcome vs response

# In[40]:


custom_palette = sns.color_palette("PRGn")
#  here we will create a pivot table for response by education and poutcome
res3 = n_df1.pivot_table(index="education", columns="poutcome", values="response_numeric")

# heatmap with percentage values
plt.figure(figsize=(10, 6))
sns.heatmap(res3 * 100, annot=True, fmt=".1f", cmap=custom_palette, center=res3.median().median() * 100)

plt.xlabel('poutcome')
plt.ylabel('Education')
plt.title('Response by Education and poutcome')

plt.show()


# # 8. Education vs marital vs response

# In[41]:


# here we will create a pivot table for response_numeric by education and marital status
res1 = n_df1.pivot_table(index="education", columns="marital", values="response_numeric", aggfunc="mean")
#print(res1.isnull().sum())
#print(res1)
# heatmap with percentage values
plt.figure(figsize=(10, 6))
sns.heatmap(res1 * 100, annot=True, fmt=".1f", cmap="Purples")

plt.xlabel('Marital Status')
plt.ylabel('Education')
plt.title('Response Rates by Education and Marital Status')

plt.show()


# # Correlation Matrix

# In[42]:


# relevant columns for correlation
selected_columns = ['age', 'balance', 'duration', 'campaign', 'month', 'previous', 'response']
corr_data = n_df1[selected_columns]

# correlation matrix
corr_matrix = corr_data.corr()

#  heatmap with annotations
plt.figure(figsize=(8, 6))
cor_plot = sns.heatmap(corr_matrix, annot=True, cmap='PiYG', linewidths=0.2, annot_kws={'size': 10})

plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Correlation Matrix', fontsize=14)

plt.show()


# Above we can see that the correlation matrix does not indicate any significant relationships between age, balance, duration, and campaign variables.

# ## Step 3. Data Visualization 

# ### 1.  Subscription and contact rate by age

# In[43]:


# creating the age groups
age_bins = [0, 29, 39, 49, 59, 100]
age_labels = ['<30', '30-39', '40-49', '50-59', '60+']

# will create a new column age_group based on age bins
n_df1['age_group'] = pd.cut(n_df1['age'], bins=age_bins, labels=age_labels)

# calculate the percentage of subscriptions by age group
subscription_by_age = n_df1[n_df1['response'] == 'yes']['age_group'].value_counts(normalize=True).sort_index() * 100

# calculate the percentage of clients contacted by age group
contacted_by_age = n_df1['age_group'].value_counts(normalize=True).sort_index() * 100

# DataFrame to store the results
age_data = pd.DataFrame({'% Subscription': subscription_by_age, '% Contacted': contacted_by_age})

# now here we will sort the DataFrame by age groups
age_data = age_data.reindex(age_labels)

age_data.plot(kind='bar', figsize=(8, 6), color=('lightblue', 'lightgreen'))
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.title('Contact Rate by Age')
plt.show()


# ### 2. Subscription by balance level

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Create balance groups based on 'balance' column
n_df1['balance_group'] = pd.cut(n_df1['balance'], bins=[-1, 0, 1000, 5000, float('inf')],
                                labels=['no balance', 'low balance', 'average balance', 'high balance'])

# Calculate subscription rate by balance group
subscription_rate = n_df1.groupby('balance_group')['response'].value_counts(normalize=True)[:, 'yes'] * 100

# Calculate the percentage of clients in each balance group
contacted_percentage = n_df1['balance_group'].value_counts(normalize=True) * 100

# Create a DataFrame
balance_df = pd.DataFrame({'% Subscription': subscription_rate, '% Contacted': contacted_percentage})

# Plot the bar chart
balance_df.plot(kind='bar', color=['purple', 'yellow'], figsize=(8, 6))
plt.title('Contact Rate by Balance Level')
plt.ylabel('Subscription Rate')
plt.xlabel('Balance Category')
plt.xticks(rotation='horizontal')

plt.show()


# ### 3. Subscription by Job

# In[45]:


import pandas as pd
import matplotlib.pyplot as plt

#subscription  by job category
subscription_rate = n_df1.groupby('job')['response'].value_counts(normalize=True)[:, 'yes'] * 100

# Sort values in ascending order
subscription_rate = subscription_rate.sort_values(ascending=True)

# Create a bar chart
job_plot = subscription_rate.plot(kind='barh', figsize=(12, 6), color="lavender")

plt.title('subscription rate by Job')
plt.xlabel('Subscription ate')
plt.ylabel('Job category')

# Label each bar
for rect, label in zip(job_plot.patches, subscription_rate.round(1).astype(str)):
    job_plot.text(rect.get_width() + 0.9, rect.get_y() + rect.get_height() - 0.5, label + '%', ha='center', va='bottom')

plt.show()


# In[46]:


# Save the DataFrame to a CSV file
file_path = '/Users/himanshutalele/Desktop/Self/Bank_Campaign_Data/output_data.csv'
delimiter = ';'
n_df1.to_csv(file_path, sep=delimiter, index=False)


# ## Target Variable:
# So from above data preprocessing and exploratory data analysis (EDA) tasks on a dataset of bank marketing campaign. 
# The "response" variable is likely the target variable for a predictive model. Hence we can use it to build a model that predicts whether a client will subscribe to a term deposit based on the available features. 

# ## Step 4.  Machine Learning
# 
# ###### Feature Engineering

# In[47]:


data = pd.read_csv('/Users/himanshutalele/Desktop/Self/Bank_Campaign_Data/output_data.csv', delimiter=';')
display(data)
data.info()


# In[48]:


data.isnull().sum()


# ##### Now we will classify which variables are related to customer and will proceed with those varibales. Also we will drop few columns which we dont need for further analysis.
# ##### So in the next step we will figure out the columns names from the cleaned dataset which is 'output_data'.

# In[49]:


data.columns


# So the Columns names are : 
# -  age
# - job
# - education
# - default
# - balance
# - housing
# - balance
# - loan

# In[50]:


# columns to keep
columns_to_keep = ['age', 'job', 'education', 'default', 'balance', 'housing', 'loan']

# here we will drop all other columns except the above ones.
df = df[columns_to_keep]
df


# Here, out of 7 columns 5 coumns contains categorical values and from those 3 contains binary values. so in the next step before proceeding train-test we will create a new Dataframe and will distribute job and education and also for binary values we will defines numberical values i.e Yes - 1 and for No - 0 

# In[51]:


# columns to one-hot encode
columns_to_encode = ['job', 'education']

# Now we will create dummy variables for the specified columns
concat_df = pd.concat([df, pd.get_dummies(df[columns_to_encode])], axis=1)

# Now we will drop the original categorical columns after creating dummy variables
concat_df.drop(columns_to_encode, axis=1, inplace=True)


# In[52]:


# List of columns to convert to binary
binary_columns = ['housing', 'default', 'loan']

# here we map those binary values columns with 1 and 0
concat_df[binary_columns] = concat_df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))


# In[53]:


concat_df.shape


# ### Data Preparation for Binary Classification
# 
# In this step, we create a clean dataset for binary classification. We start by copying the concatenated data, and then we will map the target variable 'response'- form (original dataset) to binary values within the DataFrame. The target variable 'response' will be assigned 1 if 'response' is 'yes', and 0 if 'response' is 'no'. This encoding is necessary for training binary classification models.

# In[54]:


print(data['response'].unique())


# In[55]:


data['response'] = data['response'].str.strip()


# In[56]:


response_df = pd.DataFrame(data['response'])
response_df = response_df['response'].map({'yes': 1, 'no': 0})
binary_classification_DF = pd.merge(concat_df, response_df, left_index=True, right_index=True)


# In[57]:


binary_classification_DF.info()


# ###### Feature Selection

# Now in the feature selection we will be splitting binary_classification_DF  into features (X) and the target variable (Y). Which will extracts all columns except the last one for the feature matrix X and extracts the last column for the target variable Y.

# In[58]:


binary_classification_DF = binary_classification_DF[-binary_classification_DF.age.isnull()].copy()
binary_classification_DF
#binary_classification_DF.isnull().sum()


# In[59]:


array = binary_classification_DF.values
X = array[:,0:-1] # except last column
Y = array[:,-1]


# In[60]:


# split the data into training (70) and testing (30) set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)
#binary_classification_DF


# In[69]:


# Create a DataFrame with new data
new_data = pd.DataFrame({
    'age': [43],
    'default': [1],
    'balance': [8732],
    'housing': [0],
    'loan': [1],
    'job_admin.': [0],
    'job_blue-collar': [0],
    'job_entrepreneur': [0],
    'job_housemaid': [0],
    'job_management': [1],
    'job_retired': [0],
    'job_self-employed': [0],
    'job_services': [0],
    'job_student': [0],
    'job_technician': [0],
    'job_unemployed': [0],
    'job_unknown': [0],
    'education_primary': [0],
    'education_secondary': [0],
    'education_tertiary': [1],
    'education_unknown': [0]
})


# In[70]:


# will train the logistic regression model
logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(X_train, y_train)

# predictions on the test set
y_pred = logistic_regression.predict(X_test)

# evaluation results
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# calculate ROC AUC score
y_prob = logistic_regression.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

#print("Accuracy:", accuracy)
#print("F1 Score:", f1)
#print("Precision:", precision)
#print("Recall:", recall)
#print("ROC AUC Score:", roc_auc)

results_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"],
    "Results": [accuracy, f1, precision, recall, roc_auc]
})

# results
print(results_df)

# confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


predicted_result = logistic_regression.predict(new_data)
predicted_result = ["yes" if val == 1 else "no" for val in predicted_result]

print("\nResponce:", predicted_result[0])


# In[63]:


svm_classifier = SVC(probability=True) 
svm_classifier.fit(X_train, y_train)

# predictions on the test set
y_pred = svm_classifier.predict(X_test)

# evaluation results
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# calculate ROC AUC score
y_prob = svm_classifier.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC Score"],
    "Value": [accuracy, f1, precision, recall, roc_auc]
})

print(metrics_df)

plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[64]:


naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# predictions on the test set
y_pred = naive_bayes.predict(X_test)

# evaluation results
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# calculate ROC AUC score
y_prob = naive_bayes.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC Score"],
    "Value": [accuracy, f1, precision, recall, roc_auc]
})

print(metrics_df)

plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[65]:


# will train the KNeighborsClassifier model
KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(X_train, y_train)

# predictions on the test set
y_pred = KNN.predict(X_test)

# evaluation results
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# calculate ROC AUC score
y_prob = KNN.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

results_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"],
    "Results": [accuracy, f1, precision, recall, roc_auc]
})
print(results_df)

# confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[66]:


# will train the RandomForestClassifier model
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# predictions on the test set
y_pred_rf = random_forest.predict(X_test)

# evaluation results
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)

# calculate ROC AUC score
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC Score"],
    "Value": [accuracy, f1, precision, recall, roc_auc]
})

print(metrics_df)

# confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.show()


# In[67]:


# will train the decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# predictions on the test set
y_pred_tree = decision_tree.predict(X_test)

# evaluation results
accuracy_tree = accuracy_score(y_test, y_pred_tree)
f1 = f1_score(y_test, y_pred_tree)
precision = precision_score(y_test, y_pred_tree)
recall = recall_score(y_test, y_pred_tree)

# calculate ROC AUC score
y_prob_tree = decision_tree.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC Score"],
    "Value": [accuracy, f1, precision, recall, roc_auc]
})

print(metrics_df)

# confusion matrix 
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix_tree, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Decision Tree")
plt.show()


# In[68]:


import lightgbm as lgb
from lightgbm import LGBMClassifier

# Create and train the LightGBM model
lgb_model = LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lgb = lgb_model.predict(X_test)

# Calculate evaluation metrics for LightGBM
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)

# To calculate ROC AUC score, you need probability estimates, not just predictions
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
roc_auc_lgb = roc_auc_score(y_test, y_prob_lgb)

# Create a confusion matrix for LightGBM
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)

# Create a results DataFrame for LightGBM
results_lgb = pd.DataFrame([['LightGBM', accuracy_lgb, precision_lgb, recall_lgb, f1_lgb, roc_auc_lgb]],
                            columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])

# Display the results for LightGBM
print(results_lgb)

# Display the confusion matrix for LightGBM
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix_lgb, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (LightGBM)")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




