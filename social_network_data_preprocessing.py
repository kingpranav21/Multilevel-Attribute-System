import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
# Load your attributed social network data (replace 'your_data.csv' with your actual data file)
data = pd.read_csv('your_data.csv')

# Display the first few rows of the dataset to inspect its structure
print(data.head())

# Data Preprocessing
# In this section, you can perform various preprocessing steps based on your project's requirements.
# Here are some common preprocessing tasks:

# 1. Handling Missing Data:
# Check for missing values in the dataset
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Depending on your data, you can choose to drop rows or columns with missing data or impute missing values.

# 2. Attribute Normalization or Scaling:
# If your attributes have different scales, you may want to normalize or scale them.
# Example using StandardScaler from scikit-learn:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['attribute_to_normalize'] = scaler.fit_transform(data['attribute_to_normalize'].values.reshape(-1, 1))

# 3. Encoding Categorical Variables:
# If your data includes categorical variables, you may need to one-hot encode them.
# Example:
data = pd.get_dummies(data, columns=['categorical_attribute'])

# 4. Splitting Data:
# Split your data into features (X) and the target variable (y) for machine learning.
X = data.drop(columns=['target_attribute'])
y = data['target_attribute']

# 5. Train-Test Split:
# Split the data into training and testing sets for model evaluation.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your data is now loaded, preprocessed, and split into training and testing sets, ready for model development.
def normalize_attributes(data, attributes_to_normalize):
    """
    Normalize specified attributes in the given DataFrame using StandardScaler.

    Parameters:
    - data: pandas DataFrame
      The DataFrame containing the data.
    - attributes_to_normalize: list
      A list of column names corresponding to the attributes to be normalized.

    Returns:
    - data_normalized: pandas DataFrame
      A new DataFrame with specified attributes normalized.
    """
    data_normalized = data.copy()  # Create a copy of the original data to avoid modifying it directly

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Normalize each specified attribute
    for attribute in attributes_to_normalize:
        if attribute in data_normalized.columns:
            data_normalized[attribute] = scaler.fit_transform(data_normalized[attribute].values.reshape(-1, 1))

    return data_normalized

