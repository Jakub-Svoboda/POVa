from __future__ import print_function

import cv2
import dlib
import numpy as np
import sys

# This example shows how to detect faces using dlib, localize facial landmarks,
# compute face descriptors and do face recognition (person identification).
# You shoudl be able to install all the required libraryes by:
# pip install opencv-python dlib
#
# You can use prepared python environment on merlin.fit.vutbr.cz
# export LD_LIBRARY_PATH=/mnt/matylda1/hradis/POV/3.1.0-p2.7/lib
# source /mnt/matylda1/hradis/POV/dlib/bin/activate


# Given image and face boundign box (detection) in detection,
# localize face and compute face descriptor (128-D floating-point vector).
# Documentation is at:
# faceLocalizer - http://dlib.net/python/index.html#dlib.shape_predictor
# faceNN - http://dlib.net/python/index.html#dlib.face_recognition_model_v1
# Return:
#    face descriptor - 128D numpy vector
#    licalized face shape -
def getFaceDescriptor(image, detection, faceLocalizer, faceNN):
	# FILL
	
	faceShape = faceLocalizer(image, detection)
	faceDescriptor = faceNN.compute_face_descriptor(image,faceShape)

	return faceDescriptor, faceShape


# Compute Euclidean distance between all vectors in d1 and d2.
# d1 and d2 are 2D numpy arrays where rows are the vectors to be compared.
# Then use np.argsort to get sorted indices of vectors from d2
# from the most similar to the most disimilar.
#
# Hint: Just two lines of code are needed.
#       Eucliden distance can be simply and efficiently computed as show in
#       https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
#
# Return: Matrix of sorted indices according to distances
#         with size d1_rows x d2_rows where each row contains indices
#         of vectors in d2 sorted according to their similarity to
#         a corresponding vector in d1.
def getSimilar(queryData, data):
	# FILL
	dists = -2 * np.dot(data, queryData.T) + np.sum(queryData**2, axis=1) + np.sum(data**2, axis=1)[:, np.newaxis]
	sortedMatches = np.argsort(dists)
	pass
	return sortedMatches


# Draw face bounding box and localized facial landmarks (circles with radius 4).
# The bounding box is in shape.rect
# The landmarks are in shape.parts()
# Hint: Use dir() to find out what attributes and methods these objects have.
def showFace(image, shape):
	image = image.copy()
	bb = shape.rect
	landmarks = shape.parts()

	# FILL
	image = cv2.rectangle(image, (bb.left(), bb.top()), (bb.right(), bb.bottom()), color = (255,255,255))
	for l in landmarks:
		image = cv2.circle(image, (l.x, l.y), 4,color = (255,255,255))

	cv2.imshow('Detection', image)
	cv2.waitKey(1)
	return image


def write(c):
	sys.stdout.write(c)
	sys.stdout.flush()


def markQuery(img):
	b = 20
	img = img.copy()
	img[:b, :] = np.asarray([0, 0, 255]).reshape(1, 1, 3)
	img[-b:, :] = np.asarray([0, 0, 255]).reshape(1, 1, 3)
	img[:, -b:] = np.asarray([0, 0, 255]).reshape(1, 1, 3)
	img[:, :b] = np.asarray([0, 0, 255]).reshape(1, 1, 3)
	return img


def main():

	# read all images
	images = []
	write("Reading images ")
	with open('images.txt') as f:
		for line in f:
			img = cv2.imread(line.strip())[25:-25, 25:-25, :]
			if img is not None:
				write('.')
			else:
				write('X')
			images.append(img)
	print("DONE")

	#images = images[100:110]


	# Detect faces and compute face descriptors
	faceDetector = dlib.get_frontal_face_detector()
	faceLocalizer = dlib.shape_predictor(
		'./data/shape_predictor_5_face_landmarks.dat')
	faceNN = dlib.face_recognition_model_v1(
		'./data/dlib_face_recognition_resnet_model_v1.dat')

	write("Computing face descriptors")
	faceDescriptors = np.zeros((len(images), 128), dtype=np.float32)
	for position, image in enumerate(images):
		detections = faceDetector(image[:, :, ::-1])  # Convert to RGB by ::-1
		# Don't do anything if no face detected
		if len(detections) == 0:
			write("X")
			continue
		write(".")

		# Compute descriptor for the first detected face.
		faceDescriptor, faceShape = getFaceDescriptor(
			image, detections[0], faceLocalizer, faceNN)
		faceDescriptors[position, :] = faceDescriptor

		showFace(image, faceShape)


	# Using the computed descriptors find the most similar faces.
	# Selects queryCount random images.
	# Computes distances to all other images.
	# Displays collage of resultCount the most similar images to each query.
	queryCount = 6
	resultCount = 12
	while(True):
		queryFaces = np.random.choice(len(images), queryCount)
		queryDescriptors = faceDescriptors[queryFaces]
		sortedMatches = getSimilar(queryDescriptors, faceDescriptors)

		# Get only first few best matches.
		# The first image is the query as it has zero distance to itself.
		sortedMatches = sortedMatches[:, :resultCount]

		# Create the collage.
		result = []
		for queryResult in sortedMatches:
			resultImg = [images[i] for i in queryResult]
			resultImg[0] = markQuery(resultImg[0])  # The first image is the query.
			result.append(np.concatenate(resultImg, axis=1))
		result = np.concatenate(result, axis=0)

		cv2.imshow('Result', cv2.resize(result, (0, 0), fx=0.5, fy=0.5))
		key = cv2.waitKey() & 0xFF
		if key == 27:
			break


if __name__ == "__main__":
	main()

