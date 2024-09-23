# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import plotly.graph_objects as go
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset
# df = pd.read_csv('CC General.csv')

# # Drop rows with missing values
# df = df.dropna()

# # Select relevant columns for clustering
# data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

# # Scale the data
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)

# # Perform KMeans clustering
# kmeans_model = KMeans(n_clusters=5, n_init='auto')
# clusters = kmeans_model.fit_predict(data_scaled)

# # Add clusters to the DataFrame
# df["CREDIT_CARD_SEGMENTS"] = clusters
# df["CREDIT_CARD_SEGMENTS"] = df["CREDIT_CARD_SEGMENTS"].map({
#     0: "Cluster 1",
#     1: "Cluster 2",
#     2: "Cluster 3",
#     3: "Cluster 4",
#     4: "Cluster 5"
# })

# # Set thresholds for churn
# purchase_freq_threshold = 0.1
# cash_adv_freq_threshold = 0.1
# payment_threshold = df['PAYMENTS'].quantile(0.25)  # Lower 25th percentile

# # Create the churn label
# df['CHURN'] = ((df['PURCHASES_FREQUENCY'] < purchase_freq_threshold) &
#                (df['CASH_ADVANCE_FREQUENCY'] < cash_adv_freq_threshold) &
#                (df['PAYMENTS'] < payment_threshold)).astype(int)

# # Encode categorical data if needed
# if 'CUST_ID' in df.columns:
#     # Initialize LabelEncoder
#     le = LabelEncoder()
#     # Fit and transform the 'Customer_ID' column
#     df['CUST_ID'] = le.fit_transform(df['CUST_ID'])

# # Convert other categorical columns to numeric using one-hot encoding
# df = pd.get_dummies(df)

# # Separate the features and the target variable
# X = df.drop('CHURN', axis=1)  # Features
# y = df['CHURN']               # Target variable

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the model
# rf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)

# # Calculate accuracy parameters
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print the accuracy parameters
# print("Accuracy: {:.2f}%".format(accuracy * 100))
# print("Precision: {:.2f}%".format(precision * 100))
# print("Recall: {:.2f}%".format(recall * 100))
# print("F1 Score: {:.2f}%".format(f1 * 100))

# # Print the classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Print the confusion matrix
# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Save the plot as an image
# plt.savefig('confusion_matrix.png')  # Save plot to file
# plt.close()  # Close the plot to avoid display

# # Create a 3D scatter plot for the clusters
# plot = go.Figure()

# # Loop through each unique segment and plot the data points
# for segment in df["CREDIT_CARD_SEGMENTS"].unique():
#     segment_data = df[df["CREDIT_CARD_SEGMENTS"] == segment]
#     plot.add_trace(go.Scatter3d(
#         x=segment_data['BALANCE'],
#         y=segment_data['PURCHASES'],
#         z=segment_data['CREDIT_LIMIT'],
#         mode='markers',
#         marker=dict(size=6, line=dict(width=1)),
#         name=str(segment)
#     ))

# # Update hover information
# plot.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES: %{y} <br>CREDIT_LIMIT: %{z}')

# # Update layout
# plot.update_layout(
#     width=800,
#     height=800,
#     autosize=True,
#     showlegend=True,
#     scene=dict(
#         xaxis=dict(title='BALANCE', titlefont_color='black'),
#         yaxis=dict(title='PURCHASES', titlefont_color='black'),
#         zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')
#     ),
#     font=dict(family="Gilroy", color='black', size=12)
# )

# # Save the plot as an HTML file
# plot.write_html('cluster_plot.html')

# # Reward Recommendations
# reward_recommendations = {
#     "Cluster 1": "Offer cashback on purchases and increase credit limit to encourage spending.",
#     "Cluster 2": "Provide bonus reward points for frequent purchases to maintain loyalty.",
#     "Cluster 3": "Offer balance transfer options with low interest rates to reduce balance.",
#     "Cluster 4": "Introduce rewards for increasing purchase frequency and higher payment benefits.",
#     "Cluster 5": "Give targeted offers for lowering credit utilization and increasing credit limit."
# }

# # Combine the churn prediction and clusters to tailor rewards
# df['REWARD_RECOMMENDATION'] = df.apply(
#     lambda row: reward_recommendations[row['CREDIT_CARD_SEGMENTS']] 
#     if row['CHURN'] == 0 else "Offer retention incentives like fee waivers or special discounts.", 
#     axis=1
# )

# # Display a sample of the final DataFrame with reward recommendations
# print(df[['CREDIT_CARD_SEGMENTS', 'CHURN', 'REWARD_RECOMMENDATION']].head(20))

# # Save the final DataFrame with rewards to a CSV file
# df.to_csv('customers_with_rewards.csv', index=False)
