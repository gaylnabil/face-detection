import cv2
import time
from random import randrange
from sys import platform

trained_car = cv2.CascadeClassifier('trained_classifiers/haarcascade_fullbody.xml')

capture = cv2.VideoCapture(0)
