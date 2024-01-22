import cv2
import time
from random import randrange
from sys import platform

classifier_file = 'trained_classifiers/haarcascade_fullbody.xml'

trained_car = cv2.CascadeClassifier(classifier_file)

capture = cv2.VideoCapture(0)

# used for wsl2/linux
if platform == 'linux':
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

capture.set(3, 780)
capture.set(4, 640)
color = (randrange(256), randrange(256), randrange(256))


while True:

    # Capture the video frame by frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display
    cv2.imshow('video Gray', gray)

    face_coordinates = trained_car.detectMultiScale(gray)

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

print("Camera is closed!")
