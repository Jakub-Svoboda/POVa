# coding: utf-8
from __future__ import print_function

import numpy as np
import cv2

def parseArguments():
	import argparse
	parser = argparse.ArgumentParser(
		epilog='Grab cut demonstration. ' +
		'Manually crop rectangle by mouse drag. ' +
		'An interactive Grab cut segmentation session is run on the selected crop. ' +
		'Label foreground and background pixels as needed with mouse. ' +
		'Use "f" and "b" keys to switch to foreground respective background annotation. ' +
		'Use "space" to update the segmentation. Exit by pressing Escape key.')
	parser.add_argument('-i', '--image', required=False,
						help='Image file name.', default="../du1/img.jpg")
	args = parser.parse_args()
	return args


class rectangleCropCallback(object):
	def __init__(self):
		self.firstPoint = None
		self.secondPoint = None
		self.cropping_now = False
		self.finished_cropping = False

	def mouseCallback(self, event, x, y, flags, param):
		# If the left mouse button is clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed.
		if event == cv2.EVENT_LBUTTONDOWN:
			self.firstPoint = self.secondPoint =(int(x), int(y))
			self.cropping_now = True

		# If cropping, update rectangle.
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.cropping_now == True:
				self.secondPoint = (int(x), int(y))

		# Finish cropping when left mouse button is released.
		elif event == cv2.EVENT_LBUTTONUP:
			if self.cropping_now == True:
				self.secondPoint = (int(x), int(y))
				self.cropping_now = False
				self.finished_cropping = True


class grabCutCallback(object):
	def __init__(self, image):
		self.img = image
		# Create inital foreground/background mask.
		# Lets say that all pixels probably contain background.
		# Grab cut starts from this mask and outputs results here as well.
		self.mask = np.full(
			shape=image.shape[:2], fill_value=cv2.GC_PR_BGD, dtype=np.uint8)
		self.annotating_foreground = True
		self.drawing_active = False

	def draw(self):
		img = self.img.copy()
		# Mark foreground in the image.
		img[:, :, 1][self.mask == cv2.GC_FGD] = 0
		img[:, :, 1][self.mask == cv2.GC_PR_FGD] = img[:, :, 1][self.mask == cv2.GC_PR_FGD] / 2
		# Mark background in the image.
		img[:, :, 2][self.mask == cv2.GC_BGD] = 0
		img[:, :, 2][self.mask == cv2.GC_PR_BGD] = img[:, :, 2][self.mask == cv2.GC_PR_BGD] / 2
		return img

	def mouseCallback(self, event, x, y, flags, param):
		# Start drawing into mask on mouse button press.
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing_active = True
			if self.annotating_foreground:
				self.mask[int(y), int(x)] = cv2.GC_FGD
			else:
				self.mask[int(y), int(x)] = cv2.GC_BGD

		# Draw mask pixels on mouse move.
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.drawing_active:
				if self.annotating_foreground:
					self.mask[int(y), int(x)] = cv2.GC_FGD
				else:
					self.mask[int(y), int(x)] = cv2.GC_BGD

		# Stop drawing annotation on button release.
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing_active = False


def main():
	args = parseArguments()

	# Will be using two windows.
	# One for cropping. One for segmentation.
	cv2.namedWindow('image')
	cv2.namedWindow('segmentation', cv2.WINDOW_NORMAL)

	inputImage = cv2.imread(args.image)

	# Init callback for cropping.
	cropCB = rectangleCropCallback()
	# Assign the callback to 'image' window.
	# In python cropCB.mouseCallback is a bound function
	# - it rememberes the cropCB object and can be passed as any other object.
	# FILL
	cv2.setMouseCallback('image', cropCB.mouseCallback)
	segmentCB = None

	while(True):
		# Create image copy as we will draw inside it.
		tmpImg = inputImage.copy()

		# If cropping in progress, draw the region.
		if cropCB.cropping_now:
			# Draw rectangle between cropCB.firstPoint and cropCB.secondPoint.
			# Use color (255, 0, 0). You can use cv2.rectangle().
			# Draw into tmpImg.
			# FILL
			cv2.rectangle(tmpImg, cropCB.firstPoint, cropCB.secondPoint, (255, 0, 0))

		# Start segmentation when cropping done.
		if cropCB.finished_cropping:
			cropCB.finished_cropping = False

			# Get rectangular crop.
			x1 = min(cropCB.firstPoint[0], cropCB.secondPoint[0])
			y1 = min(cropCB.firstPoint[1], cropCB.secondPoint[1])
			width = abs(cropCB.secondPoint[0] - cropCB.firstPoint[0])
			height = abs(cropCB.secondPoint[1] - cropCB.firstPoint[1])
			crop = inputImage[y1:y1 + height, x1:x1 + width, :]

			segmentCB = grabCutCallback(crop)
			# Assign the callback to 'segmentation' window.
			# FILL
			cv2.setMouseCallback("segmentation", segmentCB.mouseCallback)


		# Draw current segmentation.
		if segmentCB:
			cv2.imshow('segmentation', segmentCB.draw())

		cv2.imshow('image', tmpImg)

		key = cv2.waitKey(20) & 0xFF
		if key == 27:
			break
		elif key == ord('f') and segmentCB:
			segmentCB.annotating_foreground = True
		elif key == ord('b') and segmentCB:
			segmentCB.annotating_foreground = False
		elif key == ord(' ') and segmentCB:
			bgdModel = np.zeros((1, 65), np.float64)
			fgdModel = np.zeros((1, 65), np.float64)
			# Run cv2.grabCut() on segmentCB.img and segmentCB.mask.
			# Init with the current mask. Run 2 iterations.
			# FILL
			cv2.grabCut(segmentCB.img, 
						segmentCB.mask,
						(cropCB.firstPoint[0],cropCB.firstPoint[1], cropCB.secondPoint[0] - cropCB.firstPoint[0],	cropCB.secondPoint[1] - cropCB.firstPoint[1]),
						bgdModel,
						fgdModel,
						2)
	if segmentCB:
		# Create binary mask where True/1 is assined to cv2.GC_FGD and cv2.GC_PR_FGD.
		# cv2.cv2.GC_PR_BGD and cv2.GC_BGD are assigned False/0.
		# The source mask is in segmentCB.mask.
		# FILL
		mask = np.where((segmentCB.mask==2)|(segmentCB.mask==0),0,1).astype('uint8')

		# Adding some random foreground noise to mask.
		positions0 = np.random.random_integers(mask.shape[0] - 1, size=100)
		positions1 = np.random.random_integers(
			mask.shape[1] - 1, size=positions0.size)
		mask[positions0, positions1] = 1

		# Adding some random background noise to mask.
		positions0 = np.random.random_integers(mask.shape[0] - 1, size=100)
		positions1 = np.random.random_integers(
			mask.shape[1] - 1, size=positions0.size)
		mask[positions0, positions1] = 0

		mask = np.uint8(mask)
		cv2.imshow('noisy mask', mask * 255)

		# Remove lonely foreground pixels. Use morfological operation open -
		# erosion followed by dilatation. Use 'kernel'.
		kernel = np.ones((3, 3), dtype=np.uint8)
		# FILL
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

		# Remove small holes in foregound. Use morfological operation close -
		# dilatation followed by erosion. Use 'kernel'.
		# FILL			
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		cv2.imshow('repaired mask', np.uint8(mask) * 255)
		
		# Mask foreground pixels. Set background to 0.
		# Use 'mask' and segmentCB.img
		# FILL
		maskedForeground = segmentCB.img
		maskedForeground = cv2.bitwise_and(segmentCB.img, segmentCB.img, mask = mask)

		cv2.imshow('masked foreground', maskedForeground)

		# Distance transform - highlight pixels 20px distant from foreground
		distances = cv2.cvtColor(maskedForeground, cv2.COLOR_BGR2GRAY)
		distances = cv2.distanceTransform(distances, 2, 3, 2)

		distances = np.uint32(distances) == 20
		cv2.imshow('distance 20', np.uint8(distances) * 255)
		

		# Compute vertical and horizontal projection of the foreground.
		# Sum mask pixel in horizontal lines respective vertical columns.
		# Use matplotlib.pyplot to plot the projection graphs.
		# Use subplot() to put both graphs into a single window.
		import matplotlib.pyplot as plt

		rows = []
		pos = []
		for row in range(mask.shape[0]):
			rows.append(np.sum(mask[row]))
			pos.append(row)

		plt.subplot(2, 1, 2)	
		plt.xlabel('Rows')
		plt.plot(pos, rows)
		plt.fill_between(pos, 0, rows)
				
		
		cols = []
		pos2 = []
		cols = np.sum(mask, axis=0)
		for col in range(mask.shape[1]):
			pos2.append(col)
		plt.subplot(2, 1, 1)	
		plt.xlabel('Columns')
		plt.plot(pos2, cols)
		plt.fill_between(pos2, 0, cols)

		print(rows)
		plt.show()
		cv2.waitKey()
		

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()