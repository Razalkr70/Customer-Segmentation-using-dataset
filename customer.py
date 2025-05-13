# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# Step 3: Drop CustomerID and rename columns
df.drop('CustomerID', axis=1, inplace=True)
df.rename(columns={'Genre': 'Gender'}, inplace=True)

# Step 4: Encode Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Step 5: Feature Selection
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Step 6: Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Elbow Method to find optimal clusters
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Step 8: Fit KMeans with optimal k (say 5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 9: Visualize Clusters
sns.pairplot(df, hue='Cluster', palette='bright', diag_kind='kde')
plt.suptitle("Customer Segmentation Clusters", y=1.02)
plt.show()

# Optional: Cluster Summary
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)


cluster_labels = {
    0: "High Income, High Spend",
    1: "Low Income, Low Spend",
    2: "Young Spenders",
    3: "Savers",
    4: "Average"
}
df['Segment'] = df['Cluster'].map(cluster_labels)
print(df[[ 'Segment']].head(10))

# Step 10: 3D PCA Visualization
# Optional: 3D PCA Visualization


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df['Cluster'], cmap='rainbow')
ax.set_title("3D PCA Cluster Plot")
plt.show()
