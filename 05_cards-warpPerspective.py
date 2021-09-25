import cv2 as cv
import numpy as np

imgPath = "/home/jan/programming/python/opencv/cards.jpg"

img = cv.imread(imgPath)

width, height = 250, 350
pts1 = np.float32([[111,219],[292,188],[154,482],[352,438]])
pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])

matrix = cv.getPerspectiveTransform(pts1, pts2)
imgOutput = cv.warpPerspective(img, matrix, (width, height))

cv.imshow("cards", img)
cv.imshow("card", imgOutput)

cv.waitKey()
