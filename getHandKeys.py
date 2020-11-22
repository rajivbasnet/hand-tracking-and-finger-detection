import imutils
import cv2
import numpy as np
import random as rnd
import math
import time

def get_mask(image):
    #skin colour range
    lower_bound = np.array([0,133,77], dtype=np.uint8)
    upper_bound = np.array([235,173,127], dtype=np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)
    #removing the noise 
    kernelOpen = np.ones((4, 4))
    kernelClose = np.ones((10, 10))

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    final_mask = maskClose
    return final_mask

def get_contours(img):
    contours= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_distance(extTop, extBot):
    x1, y1 = extTop[0], extTop[1]
    x2, y2 = extBot[0], extBot[1]
    dis = math.sqrt( (x2 - x1 )**2 + (y2 -y1) **2)

    print("distance" , dis)
    return dis

def check_both_inside(extTop, extBot, rec_starting_coordinate, rec_ending_coordinate):

    is_both_inside = False
    x1, y1 = extTop[0], extTop[1]
    x2, y2 = extBot[0], extBot[1]
    recS_x, recS_y = rec_starting_coordinate[0], rec_starting_coordinate[1]
    recE_x, recE_y = rec_ending_coordinate[0], rec_ending_coordinate[1]
    if (recS_x < x1 < recE_x - 10) and (recS_y <  y1 < recE_y - 10):
        if (recS_x < x2 < recE_x) and (recS_y <  y2 < recE_y):
            is_both_inside = True
            print("BOTH POINTS INSIDE")

    return is_both_inside


def is_close_and_inside(extTop, extBot, rec_starting_coordinate, rec_ending_coordinate):
    distance = get_distance(extTop, extBot)
    if distance < 150 and check_both_inside(extTop, extBot, rec_starting_coordinate, rec_ending_coordinate):
        return True
    return False


def main():
    myCam = cv2.VideoCapture(0)
    
    WIDTH = 300
    HEIGHT = 300
    rec_starting_coordinate = (10, 10)
    rec_ending_coordinate = (rec_starting_coordinate[0] + HEIGHT, rec_starting_coordinate[0] + WIDTH)
    rec_color = (0,0,0)

    # to make displacement point relative
    prev_extTop = None

    while True:
        _, frameCap = myCam.read()
        frameCap = cv2.flip(frameCap, 1)
        frameCap = cv2.resize(frameCap, (1000, 1000))

        #rectancle 
        rec = cv2.rectangle(frameCap, rec_starting_coordinate, rec_ending_coordinate, rec_color, 4, 8, 0)

        ycr_image = cv2.cvtColor(frameCap, cv2.COLOR_BGR2YCR_CB)
        mask = get_mask(ycr_image)
        contours = get_contours(mask)

        cn = imutils.grab_contours(contours)
        maxCn = max(cn, key=cv2.contourArea)
        
        # determine the most extreme points along the contour
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])

        extTop = tuple(maxCn[maxCn[:, :, 1].argmin()][0])
        extBot = tuple(maxCn[maxCn[:, :, 0].argmin()][0])

        color = (1, 80, 225)
        cv2.drawContours(frameCap, [maxCn], -1, color, 2)

        # cv2.circle(frameCap, extLeft, 8, (0, 0, 255), -1)
        # cv2.circle(frameCap, extRight, 8, (0, 255, 0), -1)
        circle1 = cv2.circle(frameCap, extTop, 5, (255, 0, 0), -1)
        circle2 = cv2.circle(frameCap, extBot, 5, (255, 255, 255), -1)
        # show the output image
        # frameMerged = cv2.add(frameBlank, frameCap)
        disp = {}
        if is_close_and_inside(extTop, extBot, rec_starting_coordinate, rec_ending_coordinate):
            print("RECTANGLE MOVABLE: Relative to extTop and lock_extTop")
            disp['x'] = extTop[0] - prev_extTop[0] + 1
            disp['y'] = extTop[1] - prev_extTop[1] + 1
            rec_starting_coordinate = (rec_starting_coordinate[0] + disp['x'], rec_starting_coordinate[0] + disp['y'])
            rec_ending_coordinate = (rec_starting_coordinate[0] + HEIGHT, rec_starting_coordinate[0] + WIDTH)
        else:
            prev_extTop = extTop
        cv2.imshow("Draggable Object", frameCap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    myCam.release()

if __name__ == "__main__":
    main()
