# Install all the necessary libraries. These are the library imports. numpy is used for numerical computing, and cv2 is OpenCV library for computer vision tasks.
import numpy as np
import cv2
#The below line initializes the CascadeClassifier object with a pre-trained Haar classifier for face detection stored in an XML file. This classifier is capable of detecting frontal faces in an image.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#These lines read an input image called "face.jpg" and display it on the screen using the imshow function.
img = cv2.imread('face.jpg')
cv2.imshow('Original',img)
#The below line converts the original color image to grayscale. This is because grayscale images have only one channel, whereas color images have three channels (red, green, and blue), which can be computationally expensive.
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# This line uses the detectMultiScale method of the CascadeClassifier object to detect faces in the grayscale image. The method returns a list of rectangular coordinates for each face detected in the image. The arguments 1.3 and 5 control the minimum and maximum sizes of the face that the classifier will detect.
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#This loop iterates over each face detected in the image and draws a rectangle around it using the cv2.rectangle function. It also creates two new image regions of interest (ROIs), one in grayscale (roi_gray) and one in color (roi_color), for further processing.
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
#Finally, this line displays the output image with detected faces and rectangles around them on the screen using the imshow function.
cv2.imshow('Face Detected',img)
