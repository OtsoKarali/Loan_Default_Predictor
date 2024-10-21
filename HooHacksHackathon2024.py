#!/usr/bin/env python
# coding: utf-8

# # Loan Default LLM Predictor

# In[3]:


#This is a hackathon project made by Otso Karali on Saturday, October 19th, 2024.
# The dataset being used was provided from Kaggle - (https://www.kaggle.com/datasets/mishra5001/credit-card/data?select=application_data.csv)


# In[4]:


# Loading the Dataset
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('/Users/otsok/Downloads/HooHacksDataSet/application_data.csv')

# Pre-Processing

# Select the relevant columns
selected_columns = [
    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
    'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_BIRTH', 
    'CODE_GENDER', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'DAYS_REGISTRATION', 
    'TARGET'
]

# Filter the dataset to include only the selected columns
df_filtered = df[selected_columns].copy()

# Handle missing values (using median imputation for numerical columns)
imputer = SimpleImputer(strategy='median')
df_filtered['AMT_GOODS_PRICE'] = imputer.fit_transform(df_filtered[['AMT_GOODS_PRICE']])

# Function to remove outliers using IQR method
def remove_outliers_iqr(df, columns):
    df_filtered = df.copy()  # Copy to avoid changing the original data
    
    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df_filtered[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out values outside the IQR bounds
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    
    return df_filtered

# Apply IQR-based outlier removal
numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_REGISTRATION']
df_filtered_no_outliers = remove_outliers_iqr(df_filtered, numerical_cols)

# Create Debt-to-Income Ratio
df_filtered_no_outliers['Debt-to-Income Ratio'] = df_filtered_no_outliers['AMT_CREDIT'] / df_filtered_no_outliers['AMT_INCOME_TOTAL']

# For exploratory data analysis (EDA), keep the raw data without scaling
df_eda = df_filtered_no_outliers.copy()  # Copy of the raw data for EDA purposes

# Scaling the numerical columns
scaler = StandardScaler()

# Define the columns to scale
numerical_cols_to_scale = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 
                           'AMT_ANNUITY', 'Debt-to-Income Ratio', 'DAYS_BIRTH', 'DAYS_REGISTRATION']

# Create a scaled version of the data for modeling
df_scaled = df_filtered_no_outliers.copy()  # Copy for model training

# Scale numerical columns for modeling
df_scaled[numerical_cols_to_scale] = scaler.fit_transform(df_scaled[numerical_cols_to_scale])

#Proceed with splitting the data into training and testing sets

# Define target (y) and features (X)
X = df_scaled.drop('TARGET', axis=1)
y = df_scaled['TARGET']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for missing values after splitting
missing_values = df_filtered_no_outliers.isnull().sum().sort_values(ascending=False)
print("Missing values per column:\n", missing_values[missing_values > 0])

# Get basic summary statistics for the dataset
print(df_filtered_no_outliers.describe())

# Exploratory Data Analysis (EDA) example - Visualizing income distribution
plt.figure(figsize=(10,6))
sns.histplot(df_eda[df_eda['TARGET'] == 1]['AMT_INCOME_TOTAL'], color='red', label='Defaulters', kde=True)
sns.histplot(df_eda[df_eda['TARGET'] == 0]['AMT_INCOME_TOTAL'], color='blue', label='Non-Defaulters', kde=True)
plt.title('Income Distribution: Defaulters vs Non-Defaulters')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Additional EDA or model building can proceed from here

# Filter the dataset for only defaulters (TARGET == 1)
defaulters = df_filtered_no_outliers[df_filtered_no_outliers['TARGET'] == 1]

# Select only the numerical columns
numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 
                  'DAYS_BIRTH', 'DAYS_REGISTRATION', 'Debt-to-Income Ratio']

# Calculate the mean for numerical columns of defaulters
defaulters_mean = defaulters[numerical_cols].mean()

# Display the result
print("Averages for Defaulters (Numerical Columns):\n", defaulters_mean)

# For categorical columns, you can use value counts
categorical_cols = ['CODE_GENDER', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

# Display value counts for categorical columns
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(defaulters[col].value_counts())



# # Training and Testing Model

# In[5]:


# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' avoids multicollinearity

# Fit and transform the training data
X_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]), columns=encoder.get_feature_names_out())

# Remove the original categorical columns from X_train
X_train = X_train.drop(columns=categorical_cols)

# Concatenate the encoded columns back to X_train
X_train = pd.concat([X_train.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

# Do the same for X_test
X_encoded_test = pd.DataFrame(encoder.transform(X_test[categorical_cols]), columns=encoder.get_feature_names_out())

# Remove the original categorical columns from X_test
X_test = X_test.drop(columns=categorical_cols)

# Concatenate the encoded columns back to X_test
X_test = pd.concat([X_test.reset_index(drop=True), X_encoded_test.reset_index(drop=True)], axis=1)

# Step 6: Train the Random Forest Model with class weights
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=20)
rf_model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Step 9: Print results
print(f"Accuracy of the Random Forest model: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

