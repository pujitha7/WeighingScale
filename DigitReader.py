import os

import cv2
import numpy as np


def increase_brightness(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    intensity_shift = 50
    grayImage += intensity_shift
    grayImage = np.clip(grayImage, 0, 255)
    return grayImage


def numberFromROI(roiImage, fileName):
    flippedImage = cv2.flip(roiImage, -1)
    height, width = flippedImage.shape[:2]

    crop_height = (int(0.25 * height), int(0.90 * height))
    crop_width = (int(0.1 * width), int(0.92 * width))
    crop_img = flippedImage[crop_height[0]:crop_height[1], crop_width[0]:crop_width[1]]

    croppedHeight, croppedWidth = crop_img.shape[:2]
    crop_img1 = crop_img[0:croppedHeight, 0:int(0.20 * croppedWidth)]
    crop_img2 = crop_img[0:croppedHeight, int(0.25 * croppedWidth):int(0.45 * croppedWidth)]
    crop_img3 = crop_img[0:croppedHeight, int(0.50 * croppedWidth):int(0.72 * croppedWidth)]
    crop_img4 = crop_img[0:croppedHeight, int(0.75 * croppedWidth):croppedWidth]


    cv2.imshow("cropImage 1", crop_img1)
    cv2.imshow("cropImage 2", crop_img2)
    cv2.imshow("cropImage 3", crop_img3)
    cv2.imshow("cropImage 4", crop_img4)
    cv2.imshow("FlippedImage", flippedImage)
    cv2.waitKey(20000)

    cv2.imwrite("digits/1_/" + str(1) + str(fileName) + ".jpeg", crop_img1)
    cv2.imwrite("digits/2_/" + str(2) + str(fileName) + ".jpeg", crop_img2)
    cv2.imwrite("digits/3_/" + str(3) + str(fileName) + ".jpeg", crop_img3)
    cv2.imwrite("digits/4_/" + str(4) + str(fileName) + ".jpeg", crop_img4)
    cv2.imwrite("digits/5_/" + str(5) + str(fileName) + ".jpeg", flippedImage)

def load_images_from_folder(folder):
    images = []
    print(os.listdir(folder))
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            image = increase_brightness(image)
            images.append(image)
    return images


if __name__ == "__main__":
    images_data = load_images_from_folder("../../test-images")
    i = 0
    for image in images_data:
        numberFromROI(image, str(i))
        i += 1

cv2.destroyAllWindows()
