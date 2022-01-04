#!"./venv/Scripts/python.exe"

import cv2 as cv
import numpy as np

img = np.zeros((512,512,3), np.uint8)
# img[200:300, 100:300] = [255,0,0]

blue = 0
h = img.shape[0]
w = img.shape[1]
yUnit = h//255
xUnit = w//255
print(yUnit)

for x in range (0,w):
    for y in range (0,h):
        if blue > 255:
            blue = 0
        blue = 255*y/h
        img[y,x] = [int(blue), 0, 0]


cv.line(img, (10,10), (200,200), (0,255,0),1)
cv.rectangle(img, (10,10), (200,200), (0,255,0),3)
cv.circle(img, (300,300),100, (0,255,0),3)
cv.putText(img, "OPENCV", (200,200),cv.FONT_HERSHEY_COMPLEX,1, (0,255,0),1)
cv.imshow("Image", img)
print(img.shape)

cv.waitKey()