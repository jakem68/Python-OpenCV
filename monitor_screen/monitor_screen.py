import math
import cv2
import numpy as np
import pytesseract

##############################################
white = (255,255,255)
black = (0,0,0)
blue_bgr = (255,0,0)
red_bgr = (0,0,255)
green_bgr = (0,255,00)
yellow_bgr = (0,255,255)
cyan_bgr = (255,255,0)
magenta_bgr = (255,0,255)
zoom = 0.55
frameWidth = 1920
frameHeight = 1080
custom_config = r'--oem 3 --psm 6'
###############################################

cap = cv2.VideoCapture(0)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('opencv version {0}'.format(cv2.__version__))
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def mark_images(imgs, markings):
    for img, text in zip(imgs, markings):
        heigth = img.shape[0]
        width = img.shape[1]
        cv2.putText(img, text, (int(width/10), int(heigth-(heigth/10))), cv2.FONT_HERSHEY_SIMPLEX, int(heigth/200), white, int(heigth/75))
    return imgs

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
    img_blur = cv2.GaussianBlur(img, (5,5), 2)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((10,10))
    img_dilated = cv2.dilate(img_canny, kernel, iterations = 1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations = 1)
    processing_imgs.extend([img, img_gray, img_blur, img_canny, img_dilated, img_eroded])
    return processing_imgs

def get_contours(img, margin=0):
    max_area = 0
    biggest_contour_polyline = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000 :
            perim = cv2.arcLength(cnt, True)
            # print(perim)
            contourPolyline = cv2.approxPolyDP(cnt, 0.02*perim, True)
            # cv2.drawContours(img_contour, contourPolyline, -1, (0,255,255), 10)

            objCorners = (len(contourPolyline))
            # find biggest area with 4 corners
            if area > max_area and objCorners == 4:
                max_area = area
                biggest_contour_polyline = contourPolyline            
    # cv2.drawContours(img_copy, biggest_contour_polyline, -1, green_bgr, 10)
    if biggest_contour_polyline.shape == (4,1,2):
        biggest_contour_polyline = reorder(biggest_contour_polyline)
        biggest_contour_polyline[0][0][0] -= margin
        biggest_contour_polyline[0][0][1] -= margin
        biggest_contour_polyline[1][0][0] += margin
        biggest_contour_polyline[1][0][1] -= margin
        biggest_contour_polyline[2][0][0] -= margin
        biggest_contour_polyline[2][0][1] += margin
        biggest_contour_polyline[3][0][0] += margin
        biggest_contour_polyline[3][0][1] += margin
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
    hImg, wImg, _ = img.shape
    img_ratio = wImg/hImg

    # hImg = int(hImg * zoom)
    # wImg = int(wImg * zoom)
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
    margin = 0
    # img_cropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20, imgOutput.shape[2]]
    img_cropped = imgOutput[margin:imgOutput.shape[0]-margin, margin:imgOutput.shape[1]-margin]
    # img_cropped = cv2.resize(img_cropped, (warp_width, warp_height))
    return img_cropped

def detect_lines(imgs):
    linesP = cv2.HoughLinesP(imgs[5], 1, np.pi / 360, 100, None, 300, 30)
    print(len(linesP))
    print(linesP.shape)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(imgs[0], (l[0], l[1]), (l[2], l[3]), red_bgr, 3, cv2.LINE_AA)
    return imgs

def detect_words(img):
    ### detecting words
    output = pytesseract.image_to_data(img)
    # hImg, wImg, _ = img.shape
    for idx, word in enumerate(output.splitlines()):
        if idx != 0:
            data_box = word.split()
            print(data_box)
            if len(data_box) == 12:
                x0, y0, x1, y1 = int(data_box[6]), int(data_box[7]), int(data_box[8]), int(data_box[9])
                cv2.rectangle(img, (x0, y0), (x0+x1, y0+y1), green_bgr, 1)
                cv2.putText(img, data_box[11], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)
    return img

def draw_contour_img(img, contour):
    # draw contour on original image
    img_contour = img.copy()
    cv2.drawContours(img_contour, contour, -1, green_bgr, 10)
    # which contour point is which 
    if contour.shape == (4,1,2):
        for i in range(contour.shape[0]):
            cv2.putText(img_contour, str(i), (contour[i][0][0],contour[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, green_bgr, 2)
    return img_contour

def main():
    while True:
        succes, img = cap.read()

        # returns all steps in pre-processing: gray, blur, dilatated, eroded, ...
        processing_imgs = pre_processing(img)
        # returns biggest contour found in final preprocessed image and draws contour on img_contour
        contour = get_contours(processing_imgs[-1], margin = 20)
        # draw contour on original image
        img_contour = draw_contour_img(img, contour)        
        # add to image list for final representation in stack
        processing_imgs.append(img_contour)
        # get warped image, sometimes contour is empty and gives error
        if len(contour) != 0 :
            img_warped = get_warp(img, contour)
            processing_imgs.append(img_warped)
            warped = True
            cv2.imshow('cropped', img_warped)
            output = pytesseract.image_to_data(img_warped, config=custom_config)
            print(output)
        else:
            processing_imgs.append(img)
            warped = False

        markings = ['ORIG', 'GRAY', 'BLUR', 'CANNY', 'DILATED', 'ERODED', 'CONTOUR', 'WARPED']
        processing_imgs = mark_images(processing_imgs, markings)
        stacked_img = stack_images(zoom, processing_imgs, 5)
        cv2.imshow('Stack', stacked_img)

        # cv2.imshow('Result', img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
