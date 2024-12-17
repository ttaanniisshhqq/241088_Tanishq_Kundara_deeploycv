import cv2
import numpy as np
import os


def preprocess_flag(image):
    """
    Preprocess the input image by applying color segmentation and morphological cleaning.
    Handles waving flags, complex colored backgrounds, and lighting variations.
    Returns masks for red and white regions.
    """
    # Resize image for faster processing and maintain aspect ratio
    height, width = image.shape[:2]
    new_width = 500
    new_height = int(height * (new_width / width))
    resized = cv2.resize(image, (new_width, new_height))

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and white
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])  # Tighter bounds for white

    # Create masks for red and white colors
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine red masks
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Refine masks using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    return resized, red_mask, white_mask


def determine_flag_type(resized, red_mask, white_mask):
    """
    Analyze the red and white segmented regions to determine if the flag
    is Indonesian (red on top, white on bottom) or Polish (white on top, red on bottom).
    """
    # Calculate vertical center of red and white regions
    red_nonzero = np.nonzero(red_mask)
    white_nonzero = np.nonzero(white_mask)

    if len(red_nonzero[0]) == 0 or len(white_nonzero[0]) == 0:
        print("Error: Unable to detect sufficient red or white regions!")
        return resized, None

    red_center_y = np.mean(red_nonzero[0])  # Average Y coordinate of red region
    white_center_y = np.mean(white_nonzero[0])  # Average Y coordinate of white region

    # Determine flag type based on vertical positions
    if white_center_y < red_center_y:  # White is higher than red
        text = "Polish Flag: White on top, Red on bottom"
        color = (255, 255, 255)  # White text for Polish flag
    elif red_center_y < white_center_y:  # Red is higher than white
        text = "Indonesian Flag: Red on top, White on bottom"
        color = (0, 0, 255)  # Red text for Indonesian flag
    else:
        text = "Unknown Flag"
        color = (0, 255, 255)  # Yellow text for unknown flag

    # Add text to the image for visualization
    output = resized.copy()
    cv2.putText(
        output, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )

    return output, text


def main():
    # Check for file existence (either flag.jpg or flag.png)
    file_name = None
    valid_files = ["flag.jpg", "flag.png"]
    for file in valid_files:
        if os.path.exists(file):
            file_name = file
            break

    if not file_name:
        print("Error: No 'flag.jpg' or 'flag.png' found in the directory.")
        return

    # Read the image
    image = cv2.imread(file_name)
    if image is None:
        print("Error: Could not load the image.")
        return

    print("Processing the image. Please wait...")

    # Step 1: Preprocess the image to segment red and white regions
    resized, red_mask, white_mask = preprocess_flag(image)

    # Step 2: Determine the flag type based on red and white vertical positions
    output_image, flag_type = determine_flag_type(resized, red_mask, white_mask)

    # Step 3: Display only the final result with annotation
    if flag_type:
        print(f"Detected Flag: {flag_type}")
        cv2.imshow("Flag Detection Result", output_image)  # Final image with flag type

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
