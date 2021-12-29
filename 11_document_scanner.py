import math
import cv2
import numpy as np
import time

print('opencv version {0}'.format(cv2.__version__))

cap = cv2.VideoCapture(0)
print("width camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frameWidth = 1920
frameHeight = 1080
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, frameWidth)      
cap.set(4, frameHeight)

blue = [255,0,0]
red = [0,0,255]
green = [0,255,0]
orange = [0,165,255]

def waiting_time_passed(start_time, start_delay):
    time_passed = False
    if time.time() - start_time > start_delay:
        time_passed = True
    return time_passed

def stack_images(scale, imgList, cols):
    rows = math.ceil(len(imgList) / cols)
    imgArray = []
    for i in range(0, len(imgList), cols):
        imgArray.append(imgList[i:i+cols] )
    # fill up empty places with blank images
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    imageBlank = np.zeros((height, width, 3), np.uint8)
    empty_places = cols - len(imgArray[-1])
    blank_imgs = [imageBlank] * empty_places
    imgArray[-1].extend(blank_imgs)
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def pre_processing(img):
    processing_imgs = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 170, 170)
    kernel = np.ones((5,5))
    img_dilated = cv2.dilate(img_canny, kernel, iterations = 2)
    img_threshold = cv2.erode(img_dilated, kernel, iterations = 1)
    processing_imgs.extend([img, img_gray, img_blur, img_canny, img_dilated, img_threshold])
    return processing_imgs

def get_contours(img):
    max_area = 0
    biggest_contour_polyline = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000 :
            perim = cv2.arcLength(cnt, True)
            # print(perim)
            contourPolyline = cv2.approxPolyDP(cnt, 0.02*perim, True)
            objCorners = (len(contourPolyline))
            # find biggest area with 4 corners
            if area > max_area and objCorners == 4:
                max_area = area
                biggest_contour_polyline = contourPolyline            
    cv2.drawContours(img_contour, biggest_contour_polyline, -1, (0,255,0), 10)
    return biggest_contour_polyline

def reorder(my_points: np.ndarray) -> np.ndarray:
    my_points = my_points.reshape(4,2)
    new_points = np.zeros((4,1,2), np.int32)
    add = my_points.sum(1)
    # smallest of added values is equivalent of [0,0] = left-top = 0, biggest is equivalent of [FrameWidth, FrameHeight] = right-bottom = 3
    new_points[0] = my_points[np.argmin(add)]
    new_points[-1] = my_points[np.argmax(add)]
    # smallest of subtracted values is equivalent for right-top = 1, largest is equivalent of left-bottom = 2
    diff = np.diff(my_points, axis=1)
    new_points[1] = my_points[np.argmin(diff)]
    new_points[2] = my_points[np.argmax(diff)]
    # print(new_points)
    return new_points
 
def get_warp(img, contour):
    contour = reorder(contour)
    pts1 = np.float32(contour)
    pts2 = np.float32([[0,0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))
    margin = 10
    # img_cropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20, imgOutput.shape[2]]
    img_cropped = imgOutput[margin:imgOutput.shape[0]-margin, margin:imgOutput.shape[1]-margin]
    img_cropped = cv2.resize(img_cropped, (frameWidth, frameHeight))
    return img_cropped


start_time = time.time()
start_delay = 2 #sec
time_elapsed = False
while True:
    succes, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    # returns all steps in pre-processing: gray, blur, dilatated, eroded, ...
    processing_imgs = pre_processing(img)
    # returns biggest contour found in final preprocessed image and draws contour on img_contour
    img_contour = img.copy()
    contour = get_contours(processing_imgs[-1])
    # add to image list for final representation in stack
    processing_imgs.append(img_contour)
    # get warped image, sometimes contour is empty and gives error
    if len(contour) != 0 :
        warped_img = get_warp(img, contour)
        cv2.imshow('Warped', warped_img)
        processing_imgs.append(warped_img)
    else:
        processing_imgs.append(img)

    # try:
    #     processing_imgs.append(warped_img)
    # except NameError:
    #     print("warped image is none")
    #     pass

    stacked_img = stack_images(0.5, processing_imgs, 5)
    cv2.imshow('Stack', stacked_img)

    # cv2.imshow('contour', img_contour)
    
    # cv2.imshow('Result', img_contour)

    # delay_ms = 50
    # cv2.waitKey(delay_ms)

    # if time.time() - start_time > start_delay:
    #     time_elapsed = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

