# Image Processing Techniques

This document provides an overview of various image processing techniques implemented in the Streamlit application. Each section describes the method, its purpose, and any relevant mathematical formulas.

## 1. Image Enhancement Techniques

### 1.1 Noise Reduction - Non-local Means

This method reduces noise while preserving edges by averaging pixels with similar colors from different regions.

The denoising process can be described as follows:

$$
I_{\text{denoised}}(x) = \frac{1}{Z(x)} \sum_{y \in N(x)} w(x, y) I(y)
$$

where:
- \(I(x)\) is the intensity of the pixel at position \(x\),
- \(N(x)\) is a neighborhood around \(x\),
- \(w(x, y)\) is a weight that measures the similarity between pixels \(x\) and \(y\),
- \(Z(x) = \sum_{y \in N(x)} w(x, y)\) is a normalization factor.

### 1.2 Wavelet Denoising

Wavelet denoising involves transforming the image into the wavelet domain and thresholding the coefficients.

The process can be represented as:

$$
\hat{I}(x) = \sum_{j=1}^{N} \hat{c}_j \psi_j(x)
$$

where:
- \(\hat{I}(x)\) is the reconstructed image,
- \(\hat{c}_j\) are the thresholded wavelet coefficients,
- \(\psi_j(x)\) are the wavelet basis functions.

### 1.3 Histogram Equalization

This technique enhances the contrast of an image by redistributing the intensity values. The transformation is given by:

$$
s = T(r) = \sum_{i=0}^{r} \frac{n_i}{N} \cdot L
$$

where:
- \(s\) is the new intensity,
- \(r\) is the old intensity,
- \(n_i\) is the number of pixels with intensity \(i\),
- \(N\) is the total number of pixels,
- \(L\) is the maximum intensity level.

### 1.4 Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE divides the image into small regions and applies histogram equalization to each. The transformation can be represented as:

$$
T_{CLAHE}(x, y) = \frac{H_{local}(I(x,y))}{H_{global}(I(x,y))} \cdot \text{clip}(c)
$$

where:
- \(H_{local}\) is the histogram of the local region,
- \(H_{global}\) is the global histogram,
- \(c\) is the clipping limit.

## 2. Feature Detection Techniques

### 2.1 Scale-Invariant Feature Transform (SIFT)

SIFT detects features that are invariant to scale and rotation. Keypoint detection involves identifying local extrema in the difference of Gaussian (DoG):

$$
D(x, y, \sigma) = G(x, y, k \sigma) - G(x, y, \sigma)
$$

where:
- \(G\) is the Gaussian function,
- \(\sigma\) is the scale.

### 2.2 Speeded-Up Robust Features (SURF)

SURF is similar to SIFT but uses an approximation of the determinant of the Hessian matrix for keypoint detection. The determinant of the Hessian is given by:

$$
H = \begin{bmatrix}
D_{xx} & D_{xy} \\
D_{xy} & D_{yy}
\end{bmatrix}
$$

where \(D_{xx}\), \(D_{xy}\), and \(D_{yy}\) are the second-order derivatives of the Gaussian.

### 2.3 Canny Edge Detection

Canny edge detection involves several steps, including gradient calculation, non-maximum suppression, and edge tracing. The gradient magnitude is computed as:

$$
G(x, y) = \sqrt{G_x^2 + G_y^2}
$$

where \(G_x\) and \(G_y\) are the gradients in the x and y directions.

### 2.4 Blob Detection

Blob detection identifies regions in an image that differ in properties such as brightness. A common method is using the Laplacian of Gaussian (LoG):

$$
L(x, y) = \nabla^2 G(x, y, \sigma)
$$

where \(\nabla^2\) is the Laplacian operator, and \(G\) is the Gaussian function.

## 3. Segmentation Techniques

### 3.1 Otsu Thresholding

Otsu’s method automatically determines a threshold value to minimize intra-class variance:

$$
\omega_0 = \frac{N_0}{N}, \quad \omega_1 = \frac{N_1}{N}
$$

$$
\mu_0 = \frac{1}{N_0} \sum_{i=0}^{T} i \cdot P(i), \quad \mu_1 = \frac{1}{N_1} \sum_{i=T+1}^{L} i \cdot P(i)
$$

### 3.2 Sobel Edge Detection

The Sobel operator uses convolution with two kernels to compute gradients:

$$
G_x = \begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}, \quad
G_y = \begin{bmatrix}
1 & 2 & 1 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{bmatrix}
$$

The gradient magnitude is calculated as:

$$
G = \sqrt{G_x^2 + G_y^2}
$$

### 3.3 Watershed Segmentation

The watershed algorithm treats the gradient magnitude as a topographic surface and finds the "watershed lines" that separate regions. The markers are obtained by finding local minima in the gradient.

### 3.4 K-means Clustering

K-means clustering segments the image by grouping pixels based on their color values. The update steps can be expressed as:

1. Assign each pixel to the nearest centroid:
$$
c_i = \arg\min_j ||x_i - \mu_j||^2
$$

2. Update the centroids:
$$
\mu_j = \frac{1}{N_j} \sum_{i=1}^{N_j} x_i
$$

where \(N_j\) is the number of points assigned to centroid \(j\).

## 4. Registration/Normalization Techniques

### 4.1 Affine Transformation

An affine transformation can be represented as:

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} = \begin{bmatrix}
a & b & tx \\
c & d & ty \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

where \((x', y')\) are the transformed coordinates, \(a, b, c, d\) are the transformation parameters, and \((tx, ty)\) are the translation components.

### 4.2 Perspective Transformation

Perspective transformation can be expressed as:

$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix} = \begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix} \begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

The normalized coordinates are calculated as:

$$
x = \frac{x'}{w'}, \quad y = \frac{y'}{w'}
$$

where 
$$ 
\(h_{ij}\) 
$$ 
are the elements of the transformation matrix.

## Conclusion

The techniques presented in this document can be effectively used to process and analyze images. Each method has its unique strengths and is suitable for different tasks in image processing.
