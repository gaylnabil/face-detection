import cv2
import time
from random import randrange
from sys import platform
import numpy as np

def point_center(rectangle):
    (x, y, w, h) = rectangle
    center_x = x + (w // 2)
    center_y = y + (h // 2)
    return (int(center_x), int(center_y))
    
classifier_file = 'trained_classifiers/haarcascade_car.xml'

trained_car = cv2.CascadeClassifier(classifier_file)

file_name = 'videos/04.mp4'

capture = cv2.VideoCapture(file_name)
s_with = 0
s_height = 0

# used for wsl2/linux
if platform == 'linux':
    capture.set(cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'XVID'))

color = (randrange(256), randrange(256), randrange(256))

subtract = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# Configuration for detection
offset = 5
cars = []
counts = 0

while True:

    # Capture the video frame by frame
    ret, frame = capture.read()
    if ret:
        s_with = int(capture.get(3))
        s_height = int(capture.get(4))
        limit = s_height * 2 // 3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        
        # dilated = cv2.dilate(blur, np.ones((5, 5)))
        # # canny = cv2.Canny(dilated, 15, 15, 1)
        # morphology = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
        # Display
        cv2.imshow('video dilated', blur)
        
        face_coordinates = trained_car.detectMultiScale(
            blur,
            scaleFactor = 1.1,    # how much the image size is reduced at each image scale
            minNeighbors = 1,     # how many neighbors each candidate rectangle should have to retain it
        )
        
        cv2.line(frame, (0, limit), (s_with, limit), (0, 0, 255), 2) 
        cars= []
        for i, coordinate in enumerate(face_coordinates):
            # comment:
            (x, y, w, h) = coordinate
            
            if  y >= int(capture.get(4)) * 2 // 5 and w >= 100:
                print(f"coordinate [{i}] = {coordinate}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                center = point_center(coordinate)
                cars.append(center)
                
                for x, y in cars:
                    if (y < limit + offset) and (y > limit - offset):
                        cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)
                        cars.remove((x, y))
                        counts += 1
        cv2.putText(frame, f'Car Number = {counts}, limit = {limit}', (10 , 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        # end for 
        cv2.imshow('video frame', frame)
    # Check for the "q" key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()

print("Video is closed!")
