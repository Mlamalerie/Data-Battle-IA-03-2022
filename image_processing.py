import numpy as np
import cv2


def denoise_image(image: np.ndarray, method: str = 'fastNlMeansDenoising') -> np.ndarray:
    """Denoise image using OpenCV. Remove

    Args:
        image_path (str): path to image
        method (str, optional): denoising method. Defaults to 'fastNlMeansDenoising'.

    Returns:
        dst: denoised image
    """
    min_size = min(image.shape[:2])
    if method == 'fastNlMeansDenoising':
        s = int(0.04 * min_size)
        dst = cv2.fastNlMeansDenoising(image, None, s, 7, 21)
    elif method == 'bilateralFilter':
        d = 9  # Diameter of each pixel neighborhood.

        # Filter sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
        # Filter sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
        sigmaColor = sigmaSpace = 75

        dst = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    elif method == 'medianBlur':
        s = 5
        dst = cv2.medianBlur(image, s)
    elif method == 'GaussianBlur':
        s = 5
        dst = cv2.GaussianBlur(image, (s, s), 0)

    else:
        raise ValueError(f'Unknown method: {method}')
    return dst


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Equalize histogram of image using OpenCV

    Args:
        image (np.ndarray): image

    Returns:
        np.ndarray: image with equalized histogram
    """
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def binarize_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """Binarize image using OpenCV

    Args:
        image (np.ndarray): image
        method (str, optional): binarization method. Defaults to 'adaptive'.

    Returns:
        np.ndarray: binarized image
    """
    if method == 'adaptive':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f'Unknown method: {method}')
    return dst
