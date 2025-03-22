# MNIST Clustering Using Gaussian Mixture Model and Hierarchical Clustering

This project demonstrates unsupervised learning techniques to cluster the MNIST dataset using two popular clustering methods:

- **Gaussian Mixture Model (GMM)**
- **Hierarchical Clustering**

The dataset used is a small subset of the MNIST training data, which consists of grayscale images of handwritten digits (0-9).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Gaussian Mixture Model](#gaussian-mixture-model)
- [Hierarchical Clustering](#hierarchical-clustering)
- [Evaluation](#evaluation)

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mnist-clustering.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

If you are running this on Google Colab, ensure that you mount your Google Drive and update the file path to the MNIST dataset in the code:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Dataset

The MNIST dataset consists of **60,000 training images** and **10,000 test images** of handwritten digits (0-9). For this project, we use a small subset of the dataset for clustering.

You can download the MNIST dataset from [**here**](http://yann.lecun.com/exdb/mnist/) or load it using `tensorflow.keras.datasets`:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Preprocessing

Before applying clustering algorithms, we preprocess the images:

- Normalize pixel values to the range [0,1]
- Flatten the images from 28x28 to a 1D vector (784 features)
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction

```python
from sklearn.decomposition import PCA
import numpy as np

# Flatten images
x_train_flattened = x_train.reshape(len(x_train), -1) / 255.0

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
x_train_pca = pca.fit_transform(x_train_flattened)
```

## Gaussian Mixture Model

We apply the **Gaussian Mixture Model (GMM)** to cluster the images:

```python
from sklearn.mixture import GaussianMixture

# Fit GMM with 10 clusters (corresponding to 10 digits)
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(x_train_pca)
labels_gmm = gmm.predict(x_train_pca)
```

## Hierarchical Clustering

We also use **Agglomerative Hierarchical Clustering**:

```python
from sklearn.cluster import AgglomerativeClustering

# Fit Hierarchical Clustering with 10 clusters
hierarchical = AgglomerativeClustering(n_clusters=10)
labels_hierarchical = hierarchical.fit_predict(x_train_pca)
```

## Evaluation

Since clustering is an unsupervised task, we evaluate using:

- **Silhouette Score**
- **Homogeneity Score**
- **Cluster Visualization**

```python
from sklearn.metrics import silhouette_score, homogeneity_score

silhouette_gmm = silhouette_score(x_train_pca, labels_gmm)
homogeneity_gmm = homogeneity_score(y_train, labels_gmm)

silhouette_hierarchical = silhouette_score(x_train_pca, labels_hierarchical)
homogeneity_hierarchical = homogeneity_score(y_train, labels_hierarchical)

print(f"GMM - Silhouette Score: {silhouette_gmm}, Homogeneity Score: {homogeneity_gmm}")
print(f"Hierarchical - Silhouette Score: {silhouette_hierarchical}, Homogeneity Score: {homogeneity_hierarchical}")
```

## Results

The results show the effectiveness of clustering handwritten digits using unsupervised learning. Further optimizations include:

- Using **t-SNE or UMAP** for visualization
- Experimenting with different numbers of clusters
- Improving feature extraction using **autoencoders**

## License

This project is open-source and available under the **MIT License**.

## Author

[Aditya Sinha]

