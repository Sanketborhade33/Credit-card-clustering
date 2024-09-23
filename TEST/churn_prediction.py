import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\Credit retention and reward optimization using Ml\CC GENERAL.csv')

# Display the first few rows of the dataset
print(df.head())

# Set the thresholds
purchase_freq_threshold = 0.1
cash_adv_freq_threshold = 0.1
payment_threshold = df['PAYMENTS'].quantile(0.25)  # Lower 25th percentile

# Create the churn label
df['CHURN'] = ((df['PURCHASES_FREQUENCY'] < purchase_freq_threshold) &
               (df['CASH_ADVANCE_FREQUENCY'] < cash_adv_freq_threshold) &
               (df['PAYMENTS'] < payment_threshold)).astype(int)

# Save the dataset with the churn label
df.to_csv('your_dataset_with_churn_label.csv', index=False)

# Reload the dataset with the churn label
df = pd.read_csv('your_dataset_with_churn_label.csv')

# Display data types
print(df.dtypes)

# Encode categorical data if needed
if 'CUST_ID' in df.columns:
    # Initialize LabelEncoder
    le = LabelEncoder()

    # Fit and transform the 'Customer_ID' column
    df['CUST_ID'] = le.fit_transform(df['CUST_ID'])

# Convert other categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df)

# Check if there are any remaining non-numeric columns
print(df.info())

# Separate the features and the target variable
X = df.drop('CHURN', axis=1)  # Features
y = df['CHURN']               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy parameters
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the accuracy parameters
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# Save the plot as an image
plt.savefig('confusion_matrix.png')  # Save plot to file
plt.close()  # Close the plot to avoid display
