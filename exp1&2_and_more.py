
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Mount Google Drive and load the dataset
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

file_path = '/content/sample_data/mnist_train_small.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
# Assuming the first column is the labels and the rest are pixel values
X = df.drop(columns=[df.columns[0]])  # Dropping the label column
y = df[df.columns[0]]  # Labels (if needed for validation)

# Normalize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- Gaussian Mixture Model (GMM) ----------------------

# Define and train the GMM model
gmm = GaussianMixture(n_components=10, random_state=42)  # 10 for 10 digits
gmm.fit(X_scaled)

# Predict the cluster for each data point
gmm_labels = gmm.predict(X_scaled)

# Plot a few examples and their corresponding GMM cluster
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f'Cluster: {gmm_labels[i]}')
    plt.axis('off')
plt.show()

# ------------------- Hierarchical Clustering -----------------------------

# Perform hierarchical clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=10)
hierarchical_labels = hierarchical_clustering.fit_predict(X_scaled)

# Plot dendrogram (subset of 100 samples for better visualization)
Z = linkage(X_scaled[:100], 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.show()

# Plot a few examples and their corresponding Hierarchical cluster
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f'Cluster: {hierarchical_labels[i]}')
    plt.axis('off')
plt.show()

# ------------------- Evaluation (Optional) -------------------------------

# GMM Accuracy and Confusion Matrix (optional)
print("GMM Confusion Matrix:\n", confusion_matrix(y, gmm_labels))

# Hierarchical Clustering Accuracy and Confusion Matrix (optional)
print("Hierarchical Clustering Confusion Matrix:\n", confusion_matrix(y, hierarchical_labels))
