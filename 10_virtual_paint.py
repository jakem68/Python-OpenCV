import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print("width camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frameWidth = 640
frameHeight = 480

blue = [255,0,0]
red = [0,0,255]
green = [0,255,0]
orange = [0,165,255]


# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,20)

# [h_min, s_min, v_min, h_max, s_max, v_max]
no_mask_hsv = [0, 0, 0, 255, 255, 255]
orange_mask_hsv = [0, 131, 190, 60, 255, 255]
blue_mask_hsv = [87, 109, 101, 159, 255, 255]

myColorMasks = [orange_mask_hsv, blue_mask_hsv]
myColors = [orange, blue]


print('opencv version {0}'.format(cv2.__version__))

def findColor(img, myColorMasks):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_masks = []
    for color in myColorMasks:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        # imgResult = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow(str(color),mask)
        hsv_masks.append(mask)
    return hsv_masks

def showContours(masks):
    stift_tips = []
    for mask in masks:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            stift_tip = (0,0)
            area = cv2.contourArea(cnt)
            if area > 500 :
                cv2.drawContours(imgResult, cnt, -1, (0,255,0), 3)
                perim = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, perim * 0.02, True)  
                x, y, w, h = cv2.boundingRect(approx)
                stift_tip = (x+w//2, y)
                stift_tips.append(stift_tip)
    return stift_tips
            
while True:
    succes, img = cap.read()
    imgResult = img.copy()

    hsv_masks = findColor(img, myColorMasks)
    stift_tips = showContours(hsv_masks)
    for idx, stift_tip in enumerate(stift_tips):
        cv2.circle(imgResult, (stift_tip[0], stift_tip[1]), 10, myColors[idx], cv2.FILLED)


    cv2.imshow('Result', imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# succes, img = cap.read()
# cv2.imshow('Webcam', img)
# cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
