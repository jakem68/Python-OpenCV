import cv2
import pytesseract

##############################################
blue_bgr = (255,0,0)
red_bgr = (0,0,255)
green_bgr = (0,255,00)
yellow_bgr = (0,255,255)
cyan_bgr = (255,255,0)
magenta_bgr = (255,0,255)
img_path = "text_example.jpg"
###############################################

img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
custom_config = r'--oem 3 --psm 6'
# cv2.imshow('Image', img)
# cv2.waitKey(0)

output = pytesseract.image_to_string(img, config=custom_config)
print(output)

# ### detecting characters
# output = pytesseract.image_to_boxes(img)
# print(output)
# hImg, wImg, _ = img.shape
# for idx, line in enumerate(output.splitlines()):
#     char_box = line.split(' ')
#     x0, y0, x1, y1 = int(char_box[1]), int(char_box[2]), int(char_box[3]), int(char_box[4])
#     if char_box[0] == ':' or char_box[0] == 'r' or char_box[0] == 's':
#         cv2.rectangle(img, (x0,  hImg-y0), (x1, hImg-y1), green_bgr, 1)
#         cv2.putText(img, char_box[0], (x0, hImg-y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)

#     # print (x, y, w, h)
# cv2.imshow('Result', img)
# cv2.waitKey(0)

# ### detecting words
# output = pytesseract.image_to_data(img)
# hImg, wImg, _ = img.shape
# # hImg, wImg = img.shape
# for idx, word in enumerate(output.splitlines()):
#     if idx != 0:
#         data_box = word.split()
#         print(data_box)
#         if len(data_box) == 12:
#             x0, y0, x1, y1 = int(data_box[6]), int(data_box[7]), int(data_box[8]), int(data_box[9])
#             cv2.rectangle(img, (x0, y0), (x0+x1, y0+y1), green_bgr, 1)
#             cv2.putText(img, data_box[11], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)

### detecting numbers
custom_config = r'--oem 3 --psm 6 outputbase digits'
output = pytesseract.image_to_data(img, config = custom_config)
hImg, wImg, _ = img.shape
# hImg, wImg = img.shape
for idx, word in enumerate(output.splitlines()):
    if idx != 0:
        data_box = word.split()
        print(data_box)
        if len(data_box) == 12:
            x0, y0, x1, y1 = int(data_box[6]), int(data_box[7]), int(data_box[8]), int(data_box[9])
            cv2.rectangle(img, (x0, y0), (x0+x1, y0+y1), green_bgr, 1)
            cv2.putText(img, data_box[11], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)

cv2.imshow('Result', img)
cv2.waitKey(0)
