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
The MNIST dataset 
dendrogram(Z)
plt.show()
Results
The clustering results are displayed as confusion matrices, and sample images are visualized with their predicted clusters. While the clustering is unsupervised, the confusion matrix helps analyze how well the clusters correspond to the actual digit labels.
