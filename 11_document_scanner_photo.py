import math
import cv2
import numpy as np

##############################################
blue_bgr = (255,0,0)
red_bgr = (0,0,255)
green_bgr = (0,255,00)
yellow_bgr = (0,255,255)
cyan_bgr = (255,255,0)
magenta_bgr = (255,0,255)
img_path = "/home/jan/programming/python/opencv/scan_example (2).jpg"
zoom = 0.3
###############################################

print('opencv version {0}'.format(cv2.__version__))

img = cv2.imread(img_path)
print(img.shape)
hImg, wImg, _ = img.shape
img_ratio = wImg/hImg

hImg = int(hImg * zoom)
wImg = int(wImg * zoom)


# hImg, wImg = 480, 640

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
    img_canny = cv2.Canny(img_blur, 100, 100)
    kernel = np.ones((10,10))
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
        if area > 10000 :
            perim = cv2.arcLength(cnt, True)
            # print(perim)
            contourPolyline = cv2.approxPolyDP(cnt, 0.02*perim, True)
            cv2.drawContours(img_contour, contourPolyline, -1, (0,255,255), 10)

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
    # sum() sommeert een numpy matrix over een bepaalde as
    add = my_points.sum(1)
    # smallest of added values is equivalent of [0,0] = left-top = 0, biggest is equivalent of [wImg, hImg] = right-bottom = 3
    new_points[0] = my_points[np.argmin(add)]
    new_points[-1] = my_points[np.argmax(add)]
    # smallest of subtracted values is equivalent for right-top = 1, largest is equivalent of left-bottom = 2
    diff = np.diff(my_points, axis=1)
    new_points[1] = my_points[np.argmin(diff)]
    new_points[2] = my_points[np.argmax(diff)]
    # print(new_points);
    return new_points

def ratio(contour):
    temp_points = contour.reshape(4,2)
    w1 = abs(temp_points[0][0] - temp_points[1][0])
    w2 = abs(temp_points[2][0] - temp_points[3][0])
    h1 = abs(temp_points[0][1] - temp_points[2][1])
    h2 = abs(temp_points[1][1] - temp_points[3][1])
    w_avg = (w1+w2)//2
    h_avg = (h1+h2)//2
    return w_avg/h_avg

def get_warp(img, contour):
    contour = reorder(contour)
    # determine document ratio width over length
    contour_ratio = ratio(contour)
    if contour_ratio > img_ratio:
        warp_width = wImg
        warp_height = int(warp_width / contour_ratio)

    else:
        warp_height = hImg
        warp_width = int(warp_height * contour_ratio)
    pts1 = np.float32(contour)
    pts2 = np.float32([[0,0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (wImg, hImg))
    imgOutput = imgOutput[0:warp_height, 0:warp_width]
    margin = 10
    # img_cropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20, imgOutput.shape[2]]
    img_cropped = imgOutput[margin:imgOutput.shape[0]-margin, margin:imgOutput.shape[1]-margin]
    # img_cropped = cv2.resize(img_cropped, (warp_width, warp_height))
    return img_cropped


img = cv2.resize(img, (wImg, hImg))
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
    processing_imgs.append(warped_img)
else:
    processing_imgs.append(img)


stacked_img = stack_images(0.5, processing_imgs, 5)
cv2.imshow('Stack', stacked_img)
if len(contour) != 0 :
    cv2.imshow('Warped', warped_img)
cv2.waitKey(0)


