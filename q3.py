import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Helper function to display images in a grid
def show_images(images, titles, nrows=1, ncols=5):
    plt.figure(figsize=(15, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(nrows, ncols, i + 1)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Apply a High Pass Filter
def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    high_pass_img = cv2.filter2D(image, -1, kernel)
    return high_pass_img


# Apply a Low Pass Filter
def low_pass_filter(image, kernel_size=(5, 5)):
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("Kernel size must be odd! Received: {}".format(kernel_size))
    low_pass_img = cv2.GaussianBlur(image, kernel_size, 0)
    return low_pass_img


# Load two grayscale images for processing
def load_images():
    img_path1 = 'image1.jpg'
    img_path2 = 'image2.jpg'

    if not os.path.isfile(img_path1) or not os.path.isfile(img_path2):
        print("Error: One or both image paths are invalid!")
        exit()

    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Failed to load one or both images. Check image formats or paths!")
        exit()

    return img1, img2


# Combine high pass and low pass filtered images
def combine_images(high_pass_img, low_pass_img):
    if high_pass_img.shape != low_pass_img.shape:
        low_pass_img = cv2.resize(low_pass_img, (high_pass_img.shape[1], high_pass_img.shape[0]))
    combined_img = cv2.addWeighted(high_pass_img, 0.5, low_pass_img, 0.5, 0)
    return combined_img


def main():
    print("Loading images...")
    img1, img2 = load_images()
    print("Images loaded successfully!")

    print("Applying High Pass Filter on the first image...")
    high_pass_img = high_pass_filter(img1)

    print("Applying Low Pass Filter on the second image...")
    low_pass_img = low_pass_filter(img2)

    print("Combining high pass and low pass filtered images...")
    combined_img = combine_images(high_pass_img, low_pass_img)

    print("Displaying all images...")
    images = [img1, img2, high_pass_img, low_pass_img, combined_img]
    titles = ['Original Image 1', 'Original Image 2', 'High Pass Filter',
              'Low Pass Filter', 'Combined Image']

    show_images(images, titles)
    print("Process completed!")


if __name__ == "__main__":
    main()
