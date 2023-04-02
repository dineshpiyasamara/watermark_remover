# Import the necessary packages
import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('name', help='Image path')

args = parser.parse_args()


def back_remover(filename):
    # Load the image
    input_img = cv2.imread(filename)

    # Convert the image to grayscale
    gr = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # dim = gr.shape

    # Make a copy of the grayscale image
    img = gr.copy()

    # Apply morphological transformations
    for i in range(5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * i + 1, 2 * i + 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Subtract the grayscale image from its processed copy
    dif = cv2.subtract(img, gr)

    # Apply thresholding
    output_img = cv2.threshold(
        dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dark = cv2.threshold(
        output_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Extract pixels in the dark region
    darkpix = gr[np.where(dark > 0)]

    # Threshold the dark region to get the darker pixels inside it
    darkpix = cv2.threshold(
        darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Paste the extracted darker pixels in the watermark region
    output_img[np.where(dark > 0)] = darkpix.T

    output_name = "output/converted_"+os.path.basename(args.name)
    cv2.imwrite(output_name, output_img)

    input_img = cv2.resize(input_img, (700, 700))
    output_img = cv2.resize(output_img, (700, 700))

    cv2.imshow("Input image", input_img)
    cv2.imshow("Output image", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


back_remover(args.name)
