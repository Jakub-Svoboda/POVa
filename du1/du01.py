# coding: utf-8
from __future__ import print_function

import numpy as np
import cv2

# This should help with the assignment:
# * Indexing numpy arrays http://scipy-cookbook.readthedocs.io/items/Indexing.html


def parseArguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--video', help='Input video file name.')
	parser.add_argument('-i', '--image', help='Input image file name.')
	args = parser.parse_args()
	return args

	
def adjust_gamma(image, gamma=1.0): #src = https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def image(imageFileName):
	# read image
	img = cv2.imread(imageFileName)
	if img is None:
		print("Error: Unable to read image file", imageFileName)
		exit(-1)
	#showImage(img)	

	# print image width, height, and channel count
	print("Image dimensions: ", img.shape)
	
	# Resize to width 400 and height 500 with bicubic interpolation.
	img = cv2.resize(img, (400,500), interpolation = cv2.INTER_CUBIC)
	print("Image dimensions after resize: ", img.shape)
	#showImage(img)
		  
	# Print mean image color and standard deviation of each color channel
	#OpenCV uses BGR
	redImg = img[:,:,2]
	greenImg = img[:,:,1]
	blueImg = img[:,:,0]
	print("Red mean:", np.mean(np.reshape(redImg,-1)), "\t standard deviation red:", np.std(np.reshape(redImg,-1))) 
	print("Green mean:", np.mean(np.reshape(greenImg,-1)), "\t standard deviation green:", np.std(np.reshape(greenImg,-1))) 
	print("Blue mean:", np.mean(np.reshape(blueImg,-1)), "\t standard deviation blue:", np.std(np.reshape(blueImg,-1))) 

	# Fill horizontal rectangle with color 128.  
	# Position x1=50,y1=120 and size width=200, height=50
	imgRec = img.copy() # make a copy for rectangle
	imgStripes = img.copy() #and for stripes
	imgClip = img.copy() #and clip
	cv2.rectangle(imgRec, (50,120), (50+200,150+50), (128,128,128), thickness=-1)
	#showImage(imgRec)
	
	# write result to file
	cv2.imwrite('rectangle.png', imgRec)
	
	# Fill every third column in the top half of the image black.
	# The first column sould be black.  
	# The rectangle should not be visible.
	rows,cols, _ = imgStripes.shape
	for i in range(rows):
		for j in range(cols):
			if i < rows/2 and j%3 == 0:
				imgStripes[i,j] = [0,0,0]

				
	# write result to file
	cv2.imwrite('striped.png', imgStripes) 
	
	# Set all pixels with any color lower than 100 black.
	for i in range(rows):
		for j in range(cols):
			if imgClip[i,j].min() < 100:
				imgClip[i,j] = [0,0,0]	           
				#print("pixel:", imgClip[i,j]) 
	#showImage(imgClip)	

	#write result to file
	cv2.imwrite('clip.png', imgClip) ## FILL

	
def video(videoFileName):
	# open video file and get basic information
	videoCapture = cv2.VideoCapture(videoFileName)   	
	if not videoCapture.isOpened():
		print("Error: Unable to open video file for reading", videoFileName)
		exit(-1)
	frameRate = int(videoCapture.get(cv2.CAP_PROP_FPS))
	frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print("FPS:", frameRate, "width:", frame_width, "height:", frame_height)

	# open video file for writing
	videoWriter  = cv2.VideoWriter('videoOut.avi', cv2.VideoWriter_fourcc('M','J','P','G'),	frameRate, (frame_width, frame_height))        
	if not videoWriter.isOpened():
		print("Error: Unable to open video file for writing", videoFileName)
		exit(-1)
				
	while videoCapture.isOpened():
		ret, frame = videoCapture.read()
		if not ret:
			break
		
		# Flip image upside down.
		frame = cv2.flip(frame, 0)
				
		# Add white noise (normal distribution).
		# Standard deviation should be 5.
		# use np.random
		noise = np.rint(np.random.normal(0,5.0,size=(frame_height,frame_width, 3)))
		noise = np.int8(noise)
		#print(frame[0,0], noise[0,0], frame[0,0] + noise[0,0])
		#print("types: ", type(frame[0,0][0]))
		frame = np.add(frame, noise) 
		frame = np.uint8(frame)
		#print("types2: ", type(frame[0,0][0]))

		# Add gamma correction.
		# y = x^1.2 -- the image to the power of 1.2
		frame = adjust_gamma(frame, 1.2) 

		# Dim blue color to half intensity.
		redFrame = frame[:,:,2]
		greenFrame = frame[:,:,1]
		blueFrame = frame[:,:,0]
		blueFrame =  np.uint8(blueFrame*0.5)
		#print(redFrame.shape, blueFrame.shape)
		frame = np.dstack((blueFrame, greenFrame, redFrame))
				
		# Invert colors.
		frame = cv2.bitwise_not(frame)
		
		# Display the processed frame.         
		cv2.imshow("Output", frame)
		# Write the resulting frame to video file.
		videoWriter.write(frame)   
			
		# End the processing on pressing Escape.
		## FILL  
		k = cv2.waitKey(10)
		if k==27:    # Esc key to stop WORKS FOR WINDOWS, these keys are heavily platform dependant
			break
		
	cv2.destroyAllWindows()        
	videoCapture.release()
	videoWriter.release()          
	

def main():
	args = parseArguments()
	np.random.seed(1)
	image(args.image)
	video(args.video)

if __name__ == "__main__":
	main()
