MNIST Clustering Using Gaussian Mixture Model and Hierarchical Clustering
This project demonstrates unsupervised learning techniques to cluster the MNIST dataset using two popular clustering methods:

Gaussian Mixture Model (GMM)
Hierarchical Clustering
The dataset used is a small subset of the MNIST training data, which consists of grayscale images of handwritten digits (0-9).

Table of Contents
Installation
Dataset
Preprocessing
Gaussian Mixture Model
Evaluation
bash
Copy code
git clone https://github.com/your-username/mnist-clustering.git
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
If you are running this on Google Colab, ensure that you mount your Google Drive and update the file path to the MNIST dataset in the code:

python
Copy code
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
Dataset
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits. In this project, we use a small subset of the training dataset (mnist_train_small.csv), where:

The first column contains labels (0-9).
The remaining columns represent pixel values (784 features).
The dataset can be loaded from the sample_data folder in Google Colab.

Preprocessing
Feature Scaling: Before applying clustering techniques, the data is normalized using StandardScaler to standardize the pixel values.
Gaussian Mixture Model
The Gaussian Mixture Model (GMM) is a probabilistic model that assumes all data points are generated from a mixture of several Gaussian distributions.

Number of Components: We use 10 components, one for each digit (0-9).
Prediction: Each data point is assigned to one of the Gaussian components based on the probability distribution.
python
Copy code
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(X_scaled)
gmm_labels = gmm.p
dendrogram(Z)
plt.show()
Results
The clustering results are displayed as confusion matrices, and sample images are visualized with their predicted clusters. While the clustering is unsupervised, the confusion matrix helps analyze how well the clusters correspond to the actual digit labels.
