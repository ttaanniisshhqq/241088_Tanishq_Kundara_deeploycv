import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas

# Function to show an image in a subplot
def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray' if len(name.shape) == 2 else None)
    plt.title(Title)
    plt.axis('off')


# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image")
        return None
    return frame


# Function to convert a BGR image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Function for black-and-white thresholding
def threshold_black_white(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# Function to create 16 gray levels image
def threshold_grey_levels(image):
    step = 16  # Divide 0-255 into 16 grey levels
    return (image // step) * step


# Function to apply Sobel Filter
def sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Magnitude of gradients
    return np.uint8(np.absolute(sobel_combined))


# Function to apply Canny Edge Detection
def canny_edge(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)


# Function to apply Gaussian Filter
def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Manually create a Gaussian kernel and apply filtering.
    :param image: Input grayscale image
    :param kernel_size: The size of the Gaussian kernel (must be odd)
    :param sigma: The standard deviation for the Gaussian function
    :return: Filtered image
    """
    # Generate Gaussian Kernel
    kernel_half = kernel_size // 2
    x, y = np.mgrid[-kernel_half:kernel_half + 1, -kernel_half:kernel_half + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize the kernel

    # Apply Gaussian filter
    filtered_image = cv2.filter2D(image, -1, kernel)  # Apply with OpenCV
    return filtered_image


# Function to convert BGR to RGB
def convert_bgr_to_rgb(image):
    """
    Converts a BGR image to RGB color space.
    :param image: Input BGR image
    :return: RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Capture an image
image = capture_image()

if image is not None:
    # Save the captured image
    cv2.imwrite('captured_image.jpg', image)
    print("Image saved successfully.")

    # Convert the original image to grayscale
    gray_image = convert_to_grayscale(image)

    # Apply black-and-white thresholding
    threshold_image = threshold_black_white(gray_image)

    # Create 16 gray levels image
    gray_levels_image = threshold_grey_levels(gray_image)

    # Apply Sobel Filter
    sobel_image = sobel_filter(gray_image)

    # Apply Canny Edge Detection
    canny_image = canny_edge(gray_image)

    # Apply Gaussian Filter
    gaussian_filtered_image = apply_gaussian_filter(gray_image, kernel_size=5, sigma=1.0)

    # Convert BGR to RGB
    rgb_image = convert_bgr_to_rgb(image)

    # Display all images (2x4 grid)
    plt.figure(figsize=(15, 8))
    show(image, 2, 4, 1, "Original Image (BGR)")  # Display original image
    show(rgb_image, 2, 4, 2, "RGB Image")  # Display RGB converted image
    show(gray_image, 2, 4, 3, "Grayscale Image")  # Display grayscale image
    show(threshold_image, 2, 4, 4, "B/W Threshold Image")  # Black-and-white image
    show(gray_levels_image, 2, 4, 5, "16 Gray Levels Image")  # 16 gray levels
    show(sobel_image, 2, 4, 6, "Sobel Filter")  # Sobel filter result
    show(canny_image, 2, 4, 7, "Canny Edge Detector")  # Canny edge detection
    show(gaussian_filtered_image, 2, 4, 8, "Gaussian Filtered")  # Gaussian filter result

    print("Canny Edge Detection is better at edge detection than Sobel Filter.")

    plt.tight_layout()
    plt.show()
else:
    print("No image to display.")
