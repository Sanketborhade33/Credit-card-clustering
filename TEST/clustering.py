
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.io as pio

# Load the data
d = pd.read_csv('CC GENERAL.csv')

# Drop rows with missing values
d = d.dropna()

# Select relevant columns
data = d[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Perform KMeans clustering
kmeans_model = KMeans(n_clusters=5, n_init='auto')
clusters = kmeans_model.fit_predict(data_scaled)

# Add clusters to the DataFrame
a = d.copy()
a["CREDIT_CARD_SEGMENTS"] = clusters
a["CREDIT_CARD_SEGMENTS"] = a["CREDIT_CARD_SEGMENTS"].map({
    0: "Cluster 1",
    1: "Cluster 2",
    2: "Cluster 3",
    3: "Cluster 4",
    4: "Cluster 5"
})

# Create a 3D scatter plot
plot = go.Figure()

# Loop through each unique segment and plot the data points
for segment in a["CREDIT_CARD_SEGMENTS"].unique():
    segment_data = a[a["CREDIT_CARD_SEGMENTS"] == segment]
    plot.add_trace(go.Scatter3d(
        x=segment_data['BALANCE'],
        y=segment_data['PURCHASES'],
        z=segment_data['CREDIT_LIMIT'],
        mode='markers',
        marker=dict(size=6, line=dict(width=1)),
        name=str(segment)
    ))

# Update hover information
plot.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES: %{y} <br>CREDIT_LIMIT: %{z}')

# Update layout
plot.update_layout(
    width=800,
    height=800,
    autosize=True,
    showlegend=True,
    scene=dict(
        xaxis=dict(title='BALANCE', titlefont_color='black'),
        yaxis=dict(title='PURCHASES', titlefont_color='black'),
        zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')
    ),
    font=dict(family="Gilroy", color='black', size=12)
)

# Save the plot as an HTML file
plot.write_html('cluster_plot.html')
