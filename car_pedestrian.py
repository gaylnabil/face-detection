import cv2
import time
from random import randrange
from sys import platform
import numpy as np

classifier_file = 'trained_classifiers/haarcascade_car.xml'

trained_car = cv2.CascadeClassifier(classifier_file)

file_name = 'videos/03.mp4'

capture = cv2.VideoCapture(file_name)

capture.set(3, 640)
capture.set(4, 480)

# used for wsl2/linux
if platform == 'linux':
    capture.set(cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'XVID'))

color = (randrange(256), randrange(256), randrange(256))

subtract = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

while True:

    # Capture the video frame by frame
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        dilated = cv2.dilate(blur, np.ones((5, 5)))
        # canny = cv2.Canny(dilated, 15, 15, 1)
        morphology = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
        # Display
        cv2.imshow('video dilated', morphology)
        
        face_coordinates = trained_car.detectMultiScale(
            morphology,
            # scaleFactor = 1.1,    # how much the image size is reduced at each image scale
            # minNeighbors = 1,     # how many neighbors each candidate rectangle should have to retain it
        )

        for i, coordinate in enumerate(face_coordinates):
            # comment:
            (x, y, w, h) = coordinate
            print(f"coordinate [{i}] = {coordinate}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # end for

        cv2.imshow('video frame', frame)
    # Check for the "q" key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()

print("Video is closed!")
