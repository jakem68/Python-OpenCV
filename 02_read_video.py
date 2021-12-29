
import cv2

# cap = cv2.VideoCapture("C:/Users/ksj/OneDrive - Sirris/_Projecten/Alupelt_2021-1774/pictures/20210826_154353.mp4")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
print("width camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height camera is: {0}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# cap.set(3, 320)
# cap.set(4, 240)

print('opencv version {0}'.format(cv2.__version__))



while True:
    succes, img = cap.read()
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# succes, img = cap.read()
# cv2.imshow('Webcam', img)
# cv2.waitKey()

cap.release()
cv2.destroyAllWindows()

