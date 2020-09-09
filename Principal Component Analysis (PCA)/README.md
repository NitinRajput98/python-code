# Principal Component Analysis with NumPy

In this jupyter notebook, I have implemented Principal Component Analysis with NumPy in python from scratch without using any of the popular machine learning libraries such as scikit-learn and statsmodels.The aim of this implementation is to consolidate the concepts or fundamentals about PCA.

# Methodology

1.) Standardize the Data

Zero-out the mean from each feature (subtract the mean of each feature from the training set)
One must take care to preprocess the input data appropriately.
Make sure to zero-out the mean from each feature (subtract the mean of each feature from the training set), and normalize the values if your features have differing units.
PCA is best used when the data is linear, because it is projecting it onto a linear subspace spanned by the eigenvectors.
This is an important step in many machine learning algorithms, and especially so in the case of PCA. We want the PCA algorithm to give equal weight to each of the features while making the projection.
If one or more features are in a different scale than the rest, those non-standardized features will dominate the eigenvalues and give you an incorrect result This is a direct consequence of how PCA works. It is going to project our data into directions that maximize the variance along the axes. 

<img src="https://render.githubusercontent.com/render/math?math=Z = \frac{x - \mu}{\sigma}">
where,
Z	=	standard score , x	=	observed value , mu	=	mean of the sample , sigma = standard deviation of the sample


2.) Compute the Eigenvectors and Eigenvalues

There are two general ways to perform PCA. The more computationally effective way is to do something called Singular Value Decomposition or SVD.
It decomposes a matrix into the product of two unitary matrices (U, V*) and a rectangular diagonal matrix of singular values (S).
The mathematics of computing the SVD is a little complicated and out of the scope of this project. If you’re familiar with linear algebra, I highly encourage you to read about it on your own time.
Luckily for us, there is a numpy function that performs SVD on the input matrix, so we don’t really need to worry about the math for now.
We’re going to cover SVD in the next task.
Recall that PCA aims to find linearly uncorrelated orthogonal axes, which are also known as principal components (PCs) in the m dimensional space to project the data points onto those PCs. The first PC captures the largest variance in the data.
The PCs can be determined via eigen decomposition of the covariance matrix Σ. After all, the geometrical meaning of eigen decomposition is to find a new coordinate system of the eigenvectors for Σ through rotations.

Covariance: $\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^{N}(x_{ij}-\bar{x_j})(x_{ik}-\bar{x_k})$


Coviance matrix:  $Σ = \frac{1}{n-1}((X-\bar{x})^T(X-\bar{x}))$



# Data

**Link to the dataset used :** https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
