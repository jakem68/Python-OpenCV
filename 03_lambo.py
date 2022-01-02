#!"./venv/Scripts/python.exe"

import cv2 as cv
import numpy as np

imgFile = "lambo.png"
img = cv.imread(imgFile)
print(img.shape)
imgResize = cv.resize(img, (300, 200))

imgCropped = img[0:200, 200:500]

cv.imshow('Lambo', img)
cv.imshow('Lambo Resized', imgResize)
cv.imshow('Lambo Cropped', imgCropped)

cv.waitKey()
 