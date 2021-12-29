import cv2
import os

##############################################
frameWidth = 1920
frameHeight = 1080
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ocv_blue = (255,0,0)
ocv_red = (0,0,255)
ocv_green = (0,255,00)
ocv_yellow = (0,255,255)
ocv_cyan = (255,255,0)
ocv_magenta = (255,0,255)
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_russian_plate_number.xml')
numberPlateCascade = cv2.CascadeClassifier(haar_model)
minArea = 500
###############################################
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)      
cap.set(4, frameHeight)
print('opencv version {0}'.format(cv2.__version__))
print("width camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


while True:
    succes, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates = numberPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x,y,w,h) in numberPlates:
        area = w*h
        if area > minArea: 
            cv2.rectangle(img, (x,y), (x+w,y+h), ocv_magenta, 2)  
            cv2.putText(img, "Number plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, ocv_blue, 2)
            imgROI = img[y:y+h, x:x+w]


    cv2.imshow('ROI', imgROI)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# succes, img = cap.read()
# cv2.imshow('Webcam', img)
# cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
