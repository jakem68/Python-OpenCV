#!"./venv/Scripts/python.exe"

import cv2
import numpy as np
import os
from scipy import ndimage

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
print(" ")
print(haar_model)

blue = (255,0,0)
red = (0,0,255)
green = (0,255,00)

faceCascade = cv2.CascadeClassifier(haar_model)
# imgFile = "/home/jan/programming/python/opencv/Lenna.png"
imgFile = "/home/jan/programming/python/opencv/people.jpg"
img = cv2.imread(imgFile)

#rotation angle in degree
# img = ndimage.rotate(img, 45)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), blue, 2)  

cv2.imshow('Gray', imgGray)
cv2.imshow('Result', img)

cv2.waitKey()
