import cv2, os
import numpy as np
import keras.preprocessing.image as image


# binarization function
def binarization(img):
	# 89 is a approximation reference from paper.
	# originally in paper it mentions 0.35 as the threshold for binarization
	_, thresh1 = cv2.threshold(img, 89, 255, cv2.THRESH_BINARY)
	return thresh1

# Dataloader
def path_reader(path):
	list_of_path = []
	for root, folders, files in os.walk(path):
		for file in files:
			list_of_path += [root + file]
	return list_of_path

# read image
def read_images_from_path(path_list):
	imgs_list = []
	for path in path_list:
		img = image.load_img(path, grayscale=False, color_mode="rgb", target_size=None, interpolation="nearest")
		imgs_list += [img]
	imgs_list = np.vstack(imgs_list)
	return imgs_list	