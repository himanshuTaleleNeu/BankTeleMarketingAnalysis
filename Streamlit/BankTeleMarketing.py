# %%
import sys
sys.executable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from IPython.display import display

# %%
data = pd.read_csv('https://raw.githubusercontent.com/himanshuTaleleNeu/BankTeleMarketingAnalysis/main/Streamlit/output_data.csv')
# data = pd.read_csv('/Users/himanshutalele/Desktop/Developer/BankTeleMarketingAnalysis/Streamlit/output_data.csv', delimiter=';')
# df = pd.DataFrame(data)
csv = pd.read_csv(data, delimiter=';')
df = pd.DataFrame(csv)

# %%
# columns to keep
columns_to_keep = ['age', 'job', 'education', 'default', 'balance', 'housing', 'loan']

# here we will drop all other columns except the above ones.
df = df[columns_to_keep]
df

# %%
# columns to one-hot encode
columns_to_encode = ['job', 'education']

# Now we will create dummy variables for the specified columns
concat_df = pd.concat([df, pd.get_dummies(df[columns_to_encode])], axis=1)

# Now we will drop the original categorical columns after creating dummy variables
concat_df.drop(columns_to_encode, axis=1, inplace=True)
print(concat_df.columns)


# %%
binary_encoded_columns = ['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
                          'job_other', 'job_retired', 'job_self-employed', 'job_services', 'job_student',
                          'job_technician', 'job_unemployed', 'education_other', 'education_primary',
                          'education_secondary', 'education_tertiary']



# Convert 'False' and 'True' to 0 and 1 in the specified columns
concat_df[binary_encoded_columns] = concat_df[binary_encoded_columns].astype(int)
print(concat_df.columns)

# %%
# List of columns to convert to binary
binary_columns = ['housing', 'default', 'loan']

# here we map those binary values columns with 1 and 0
concat_df[binary_columns] = concat_df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

concat_df.shape

print(data['response'].unique())

data['response'] = data['response'].str.strip()

response_df = pd.DataFrame(data['response'])
response_df = response_df['response'].map({'yes': 1, 'no': 0})
binary_classification_DF = pd.merge(concat_df, response_df, left_index=True, right_index=True)

binary_classification_DF.info()

binary_classification_DF = binary_classification_DF[-binary_classification_DF.age.isnull()].copy()
binary_classification_DF
binary_classification_DF.isnull().sum()

# %%
array = binary_classification_DF.values
# X = array[:,0:-1] # except last column
# Y = array[:,-1]

X = binary_classification_DF.drop(columns=['response']) 
Y = binary_classification_DF['response']
# split the data into training (70) and testing (30) set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)
#binary_classification_DF

# will train the logistic regression model
logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(X_train, y_train)

# predictions on the test set
y_pred = logistic_regression.predict(X_test)

# evaluation results
accuracy_logistic_regression = accuracy_score(y_test, y_pred)


results_df = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Results": [accuracy_logistic_regression]
})

# results
print(results_df)


def predict_response(age, default, balance, housing, loan, job, education):
    # mappings for job and education columns
    job_mapping = {
        'admin.': 0,
        'blue-collar': 1,
        'entrepreneur': 2,
        'housemaid': 3,
        'management': 4,
        'retired': 5,
        'self-employed': 6,
        'services': 7,
        'student': 8,
        'technician': 9,
        'unemployed': 10,
        'unknown': 11
    }
    
    education_mapping = {
        'primary': 0,
        'secondary': 1,
        'tertiary': 2,
        'unknown': 3
    }

    # Create arrays to represent job and education values
    job_values = np.zeros(12)  # Initialize with zeros
    education_values = np.zeros(4)  # Initialize with zeros

    # Ensure job and education values are valid
    if job in job_mapping and education in education_mapping:
        job_values[job_mapping[job]] = 1  # Set the corresponding job value to 1
        education_values[education_mapping[education]] = 1  # Set the corresponding education value to 1
    else:
        return "Invalid job or education value"

    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'age': [age],
        'default': [default], 
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'job_admin.': job_values[0],
        'job_blue-collar': job_values[1],
        'job_entrepreneur': job_values[2],
        'job_housemaid': job_values[3],
        'job_management': job_values[4],
        'job_other': job_values[5],
        'job_retired': job_values[6],
        'job_self-employed': job_values[7],
        'job_services': job_values[8],
        'job_student': job_values[9],
        'job_technician': job_values[10],
        'job_unemployed': job_values[11],
        'education_other': education_values[0],
        'education_primary': education_values[1],
        'education_secondary': education_values[2],
        'education_tertiary': education_values[3],
    })

    # Make predictions using the logistic regression model
    predicted_result = logistic_regression.predict(new_data)
    
    return "yes" if predicted_result[0] == 1 else "no"
