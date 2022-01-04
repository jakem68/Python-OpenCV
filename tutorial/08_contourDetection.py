#!"./venv/Scripts/python.exe"

import cv2 as cv
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getContours(imagOrig, imgCanny):
    imgContour = imgOrig.copy()
    contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500 :
            cv.drawContours(imgContour, cnt, -1, (0,255,0), 3)
            perim = cv.arcLength(cnt, True)
            # print(perim)
            contourPolyline = cv.approxPolyDP(cnt, 0.02*perim, True)
            objCorners = (len(contourPolyline))
            x, y, w, h = cv.boundingRect(contourPolyline)
            if objCorners == 3:
                objectType = "Triangle"
            elif objCorners == 4:
                aspectRatio = w/float(h)
                print(aspectRatio)
                if aspectRatio > 0.95 and aspectRatio < 1.05 :
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCorners > 4 :
                objectType = "Circle"
            else:
                objectType = "None"


            cv.rectangle(imgContour, (x, y),(x+w, y+h), (0,0,255))
            cv.putText(imgContour, objectType, (x+(w//2)-10, y+(h//2)-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

        print(area)
    return imgContour



imgFile = "shapes.png"
imgOrig = cv.imread(imgFile)
imgGray = cv.cvtColor(imgOrig, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv.Canny(imgBlur, 50, 50)
imgBlank = np.zeros_like(imgOrig)

imgContour = getContours(imgOrig, imgCanny)

imgStack = stackImages(0.8, ([imgOrig, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))

# cv.imshow('Original', img)
# cv.imshow('Grey', imgGray)
cv.imshow('Stacked', imgStack)

# if cv.waitKey(1) & 0xFF == ord('q'):
#     break

cv.waitKey() 
# cv.destroyAllWindows()
