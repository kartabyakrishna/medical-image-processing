import streamlit as st
import cv2
import numpy as np
from skimage import filters, feature, segmentation, color
from skimage.restoration import denoise_wavelet
from skimage.filters import threshold_otsu, sobel

# Define Image Enhancement Techniques
def image_enhancement(img, method):
    if method == 'Noise Reduction - Non-local Means':
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif method == 'Wavelet Denoising':
        return denoise_wavelet(img, multichannel=True)
    elif method == 'Histogram Equalization':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(img_gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    elif method == 'Contrast Limited AHE':
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img

# Define Feature Detection Techniques
def feature_detection(img, method):
    if method == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, keypoints, None)
    elif method == 'SURF':
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, descriptors = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, keypoints, None)
    elif method == 'Canny Edge Detection':
        edges = cv2.Canny(img, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif method == 'Blob Detection':
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(img)
        img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# Define Segmentation Techniques
def segmentation(img, method):
    if method == 'Otsu Thresholding':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif method == 'Sobel Edge Detection':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = sobel(gray)
        return cv2.cvtColor((edges * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif method == 'Watershed Segmentation':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(binary, sure_fg)
        markers = cv2.connectedComponents(sure_fg)[1]
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]
    elif method == 'K-means Clustering':
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img = res.reshape((img.shape))
    return img

# Define Registration/Normalization Techniques
def registration_and_normalization(img, method):
    if method == 'Affine Transformation':
        rows, cols, ch = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (cols, rows))
    elif method == 'Perspective Transformation':
        rows, cols, ch = img.shape
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (300, 300))
    return img

# Streamlit UI
st.title("Interactive Image Processing App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Image Enhancement Options
    enhancement_method = st.selectbox("Select Image Enhancement Technique", ["None", "Noise Reduction - Non-local Means", "Wavelet Denoising", "Histogram Equalization", "Contrast Limited AHE"])
    if enhancement_method != "None":
        img = image_enhancement(img, enhancement_method)
        st.image(img, caption=f"Enhanced Image - {enhancement_method}")

    # Feature Detection Options
    feature_method = st.selectbox("Select Feature Detection Technique", ["None", "SIFT", "SURF", "Canny Edge Detection", "Blob Detection"])
    if feature_method != "None":
        img = feature_detection(img, feature_method)
        st.image(img, caption=f"Feature Detection - {feature_method}")

    # Segmentation Options
    segmentation_method = st.selectbox("Select Segmentation Technique", ["None", "Otsu Thresholding", "Sobel Edge Detection", "Watershed Segmentation", "K-means Clustering"])
    if segmentation_method != "None":
        img = segmentation(img, segmentation_method)
        st.image(img, caption=f"Segmented Image - {segmentation_method}")

    # Registration and Normalization Options
    registration_method = st.selectbox("Select Registration/Normalization Technique", ["None", "Affine Transformation", "Perspective Transformation"])
    if registration_method != "None":
        img = registration_and_normalization(img, registration_method)
        st.image(img, caption=f"Transformed Image - {registration_method}")
