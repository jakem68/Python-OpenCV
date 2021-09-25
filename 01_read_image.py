#!"./venv/Scripts/python.exe"

import cv2 as cv
import numpy as np

print('opencv version {0}'.format(cv.__version__))

kernel = np.ones((5,5), np.uint8)

imgFile = "/home/jan/programming/python/opencv/Lenna.png"
img = cv.imread(imgFile)
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (7,7), 0)
imgCanny1 = cv.Canny(img, 100, 100)
imgCanny = cv.Canny(img, 150, 200)

imgDilation = cv.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv.erode(imgDilation, kernel, iterations=1)

cv.imshow('Lenna', img)
cv.imshow('LennaGray', imgGray)
cv.imshow('LennaGrayBlur', imgBlur)
cv.imshow('LennaCanny1', imgCanny1)
cv.imshow('LennaCanny', imgCanny)

cv.imshow('imgDilated', imgDilation)
cv.imshow('imgEroded', imgEroded)

cv.waitKey()
