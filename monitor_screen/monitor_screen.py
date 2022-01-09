

__author__ = 'Jan Kempeneers'


import math, cv2, time, pytesseract, json
import numpy as np
import pandas as pd
from my_mqtt_module import Mqtt


##############################################
white = (255,255,255)
black = (0,0,0)
blue_bgr = (255,0,0)
red_bgr = (0,0,255)
green_bgr = (0,255,00)
yellow_bgr = (0,255,255)
cyan_bgr = (255,255,0)
magenta_bgr = (255,0,255)
custom_config = r'--oem 3 --psm 6'
###############################################


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
    # img_bright = img_gray.copy()
    # cv2.normalize(img_bright, img_bright, -100, 355, cv2.NORM_MINMAX)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 4)
    # img_canny = cv2.Canny(img_blur, 50, 50)
    img_canny = cv2.Canny(img_blur, 100, 50)
    kernel = np.ones((3,3))
    img_dilated = cv2.dilate(img_canny, kernel, iterations = 2)
    img_eroded = cv2.erode(img_dilated, kernel, iterations = 3)
    processing_imgs.extend([img, img_gray, img_blur, img_canny, img_dilated, img_eroded])
    markings = ['ORIG', 'GRAY', 'BLUR', 'CANNY', 'DILATED', 'ERODED', 'CONTOUR', 'WARPED']

    return processing_imgs, markings

def get_contours(img, margin=0):
    contour = False
    max_area = 0
    biggest_contour_polyline = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000 :
            perim = cv2.arcLength(cnt, True)
            contourPolyline = cv2.approxPolyDP(cnt, 0.02*perim, True)
            # cv2.drawContours(img_contour, contourPolyline, -1, (0,255,255), 10)

            objCorners = (len(contourPolyline))
            # find biggest area with 4 corners
            if area > max_area and objCorners == 4:
                max_area = area
                biggest_contour_polyline = contourPolyline            
    # cv2.drawContours(img_copy, biggest_contour_polyline, -1, green_bgr, 10)
    if biggest_contour_polyline.shape == (4,1,2):
        contour = True
        biggest_contour_polyline = reorder(biggest_contour_polyline)
        biggest_contour_polyline[0][0][0] -= margin
        biggest_contour_polyline[0][0][1] -= margin
        biggest_contour_polyline[1][0][0] += margin
        biggest_contour_polyline[1][0][1] -= margin
        biggest_contour_polyline[2][0][0] -= margin
        biggest_contour_polyline[2][0][1] += margin
        biggest_contour_polyline[3][0][0] += margin
        biggest_contour_polyline[3][0][1] += margin
    return biggest_contour_polyline, contour

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

# def detect_lines(imgs):
#     linesP = cv2.HoughLinesP(imgs[5], 1, np.pi / 360, 100, None, 300, 30)
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             cv2.line(imgs[0], (l[0], l[1]), (l[2], l[3]), red_bgr, 3, cv2.LINE_AA)
#     return imgs

# def detect_words(img):
#     ### detecting words
#     output = pytesseract.image_to_data(img)
#     # hImg, wImg, _ = img.shape
#     for idx, word in enumerate(output.splitlines()):
#         if idx != 0:
#             data_box = word.split()
#             if len(data_box) == 12:
#                 x0, y0, x1, y1 = int(data_box[6]), int(data_box[7]), int(data_box[8]), int(data_box[9])
#                 cv2.rectangle(img, (x0, y0), (x0+x1, y0+y1), green_bgr, 1)
#                 cv2.putText(img, data_box[11], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)
#     return img

def draw_contour_img(img, contour):
    # draw contour on original image
    img_contour = img.copy()
    cv2.drawContours(img_contour, contour, -1, green_bgr, 10)
    # which contour point is which 
    for i in range(contour.shape[0]):
        cv2.putText(img_contour, str(i), (contour[i][0][0],contour[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, green_bgr, 2)
    return img_contour

# def get_boxes_and_words(df: pd.DataFrame):
#     # for idx, row in df.iterrows():
#     # We drop some redundant columns
#     df.drop(['level', 'page_num', 'block_num', 'par_num'], axis=1, inplace=True)
#     # drop rows where confident < 50
#     df.drop(df[df['conf'] < 50].index, inplace=True)
#     boxes = list(zip(df.left.tolist(), df.top.tolist(), df.width.tolist(), df.height.tolist(), df.text.tolist()))
#     # df['lines'] = df.groupby(['line_num'])['text'].transform(lambda x : ' '.join(x))
#     # df.drop_duplicates("lines",keep = 'last', inplace = True)
#     return boxes

# def draw_boxes(boxes, img):
#     for box in boxes:
#         cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), green_bgr, 1)
#         cv2.putText(img, box[4], (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_bgr)

def crop_and_read_fields(contour, processing_imgs):
    img = processing_imgs[0]
    img_contour = draw_contour_img(img, contour)        
    # add to image list for final representation in stack
    processing_imgs.append(img_contour)
    # get warped image, sometimes contour is empty and gives error
    # if len(contour) != 0 :
    img_warped = get_warp(img, contour)
    processing_imgs.append(img_warped)
    
    # # First attempt
    # output = pytesseract.image_to_data(img_warped, config=custom_config, output_type='data.frame')
    # data_blocks = get_boxes_and_words(output)
    # draw_boxes(data_blocks, img_warped)

    # Second attempt
    # don't read all text from warped image at once but define 
    # new fixed ROIs for each value based on warped image

    # list of ROI coordinates, manually taken of the screen from img_warped
    rois = {'speed':((115,95),(175,118)), 'feed':((115,119),(175,141)), 
            'tool':((115,142), (175,165)), 'status':((413,140),(550,170)), 
            'error messages':((413,180),(550,210))}
    scan_results = {}
    for key in rois:
        x0 = rois[key][0][0]
        x1 = rois[key][1][0]
        y0 = rois[key][0][1]
        y1 = rois[key][1][1]
        imgt = img_warped[y0:y1, x0:x1]
        #### prepare img for tesseract
        # make gray
        imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
        # resize
        imgt = cv2.resize(imgt, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        # sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        imgt = cv2.filter2D(src=imgt, ddepth=-1, kernel=kernel)
        # make black and white
        (thresh, imgt) = cv2.threshold(imgt, 127, 255, cv2.THRESH_TRIANGLE)
        # dilate
        kernel = np.ones((4, 4), np.uint8)
        imgt = cv2.dilate(imgt, kernel, iterations=1)
        # for testing
        # cv2.imshow('temp cropped', imgt)
        # cv2.waitKey(0)
        ## handle output from tessereact
        output = pytesseract.image_to_string(imgt, config=custom_config)
        output = output.lower().strip()
        scan_results[key]=output
        cv2.rectangle(img_warped, (x0, y0), (x1, y1), green_bgr, 1)
        result_str = f'{key} is {output}'
        cv2.putText(img_warped, result_str, (x0+3, y1-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, blue_bgr)    
    cv2.imshow('cropped', img_warped)
    return processing_imgs, scan_results

def evaluate_scan_list(scan_results):
    df_results = pd.DataFrame(scan_results)
    scan_evaluated = []
    for column in df_results:
        value = df_results[column].value_counts()[:1].index.tolist()[0]
        value = value.lower()
        count = df_results[column].value_counts()[:1].tolist()[0]
        column_length = df_results.shape[0]
        certainty = count / column_length
        result = {'topic': column, 'value': value, 'certainty': certainty}
        if float(result['certainty']) < 0.6:
            result['value'] = 'noread'
        scan_evaluated.append(result)
    return scan_evaluated

def detect_changes_between_evaluated_scans(scan1, scan2):
    change_occurred = False
    for dict1, dict2 in zip(scan1, scan2):
        if dict1['value'] != dict2['value']:
            change_occurred = True
    return change_occurred

def update_message(payload):
    message = {"timestamp":"", "payload": ""}
    payload_str = json.dumps(payload)
    message["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
    message["payload"] = payload_str
    msg_out = json.dumps(message)
    return msg_out

def run():
    # my_mqtt = Mqtt("/media/usb/MFRC522-python/rfid_to_mosquitto.yml")
    mqtt_active = True
    if mqtt_active:
        my_mqtt = Mqtt("/home/jan/programming/python/opencv/monitor_screen/monitor_to_mosquitto.yml")
        my_mqtt.start()
        mqtt_start_time = time.time()

    zoom = 0.55
    # frameWidth = 1920
    # frameHeight = 1080

    cap = cv2.VideoCapture(0)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('opencv version {0}'.format(cv2.__version__))
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    print (f'expose is {cap.get(cv2.CAP_PROP_EXPOSURE)}')
    scan_results = []
    scans_evaluated_before = None
    while True:
        succes, img = cap.read()
        # returns all steps in pre-processing: gray, blur, dilatated, eroded, ...
        processing_imgs, markings = pre_processing(img)
        # returns biggest contour found in final preprocessed image and draws contour on img_contour
        contour, contour_detected = get_contours(processing_imgs[-1], margin = 5) # contour is numpy ndarray shape (4,1,2)

        # draw contour on original image
        cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cropped", 1600, 800)
        if contour_detected:
            processing_imgs, scan_result = crop_and_read_fields(contour, processing_imgs )
            # scan_result example : 
            # {'speed': '100%', 'feed': '80%', 'tool': '73.1', 'status': 'Idle', 'error messages': 'No issues'}
            scan_results.append(scan_result) 

        else:
            processing_imgs.append(img.copy())
            if scan_results:
                # scan_results.pop(0)
                scan_results = []
            if cv2.getWindowProperty('cropped', 0) >= 0:
                cv2.destroyWindow("cropped")

        processing_imgs = mark_images(processing_imgs, markings)
        stacked_img = stack_images(zoom, processing_imgs, 5)
        cv2.imshow('Stack', stacked_img)

        # start evaluating scan results after 10 good scans
        change_detected = False
        if len(scan_results) == 10:
            # filters the topmost found values and their respective relative occurence in 10 scans
            scans_evaluated = evaluate_scan_list(scan_results)
            # example scans_evaluated
            # [{'topic': 'speed', 'value': '100%', 'certainty': 0.8}, 
            # {'topic': 'feed', 'value': '80%', 'certainty': 0.8}, 
            # {'topic': 'tool', 'value': '73.1', 'certainty': 0.7}, 
            # {'topic': 'status', 'value': 'Idle', 'certainty': 0.8}, 
            # {'topic': 'error messages', 'value': 'No issues', 'certainty': 0.7}]
            if scans_evaluated_before != None:
                change_detected = detect_changes_between_evaluated_scans(scans_evaluated, scans_evaluated_before)
            else:
                change_detected = True
            scans_evaluated_before = scans_evaluated
            # scan_results.pop(0)
            scan_results = []

        # if change_detected send mqtt message of evaluated scans
        if change_detected:
            print (scans_evaluated)
            msg_out = update_message(scans_evaluated)
            if mqtt_active:
                my_mqtt.publish(msg_out)

        if mqtt_active:
            if time.time() - mqtt_start_time > 30:
                my_mqtt.client.disconnect()
                mqtt_start_time = time.time()
                my_mqtt.start()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # output.to_csv("tesseract.csv", index = False)

def main():
    run()

if __name__ == "__main__":
    main()
