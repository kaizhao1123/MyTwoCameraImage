import cv2 as cv
import numpy as np
import math


########################################################################
# normalized img_top to the same size as img_side (decrease)
########################################################################
def normalizeImage(img_side, img_top, HSV_Lower_side, HSV_Lower_top, HSV_Upper):

    contour_side = 0
    contour_top = 0
    weightAndHeight = []

    ##
    #  get the contour in img_side
    ##

    # convert the image to the hsv data format
    hsv = cv.cvtColor(img_side, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (3, 3), 10)
    hsv = cv.dilate(hsv, (3, 3))
    # Threshold the HSV image to get only brown colors
    mask = cv.inRange(hsv, HSV_Lower_side, HSV_Upper)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img_side, [box], 0, (0, 0, 255), 2)  # red
        contour_side = cont
        weightAndHeight.append(rect[1])     # length and height

    ##
    #  get the contour in img_top
    ##

    # convert the image to the hsv data format
    hsv = cv.cvtColor(img_top, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (3, 3), 10)
    hsv = cv.dilate(hsv, (3, 3))
    # Threshold the HSV image to get only brown colors
    mask = cv.inRange(hsv, HSV_Lower_top, HSV_Upper)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # cv.drawContours(img_top, [box], 0, (0, 0, 255), 2)  # red
        contour_top = cont
        # weightAndHeight.append(rect[1])     # length and width

    ##
    #   normalize img_top to img_side, and get parameters
    ##

    x, y, w, h = cv.boundingRect(contour_side)   # side
    x_t, y_t, w_t, h_t = cv.boundingRect(contour_top)   # top
    print(x, y, w, h)
    print(x_t, y_t, w_t, h_t)
    r = w_t / w
    targetImg = img_top[y_t:y_t+h_t, x_t:x_t+w_t]
    newW = w   # w_t / r
    newH = math.floor(h_t / r)
    dim = (newW, newH)
    newTarget = cv.resize(targetImg, dim, interpolation=cv.INTER_AREA)
    template = np.zeros([newH + 30, newW + 30, 3], dtype=np.uint8)
    template[15: newH + 15, 15: newW + 15] = newTarget

    # img_top[y_t-15: y_t+15 + newH, x_t-15: x_t+15 + newW] = template  # for the real top view

    img_top[265 - 30 - newH: 265, x_t - 15: x_t + 15 + newW] = template        # for the assumed top view

    # convert the image to the hsv data format
    hsv = cv.cvtColor(img_top, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (3, 3), 10)
    hsv = cv.dilate(hsv, (3, 3))
    # Threshold the HSV image to get only brown colors
    mask = cv.inRange(hsv, HSV_Lower_top + 0, HSV_Upper)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    for cont in contours:
        area = cv.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:
            continue
        rect = cv.minAreaRect(cont)

        # draw the min area rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img_top, [box], 0, (0, 0, 255), 2)  # red
        contour_top = cont
        weightAndHeight.append(rect[1])     # length and width

    # print(weightAndHeight)
    return contour_side, contour_top


########################################################################
# get left and right most point
########################################################################
def getLeftRight(contour_dict):
    listKey = list(contour_dict)
    leftKey = listKey[0]
    leftValue = contour_dict.get(leftKey)
    left_Y = int(sum(leftValue) / len(leftValue))
    leftPoint = (leftKey, left_Y)

    rightKey = listKey[-1]
    rightValue = contour_dict.get(rightKey)
    right_Y = int(sum(rightValue) / len(rightValue))
    rightPoint = (rightKey, right_Y)

    return leftPoint, rightPoint


########################################################################
# getMidPoint function return the mid point
########################################################################
def getMidPoint(pointl, pointr, x):
    a = float((pointr[1] - pointl[1]) / (pointr[0] - pointl[0]))
    b = pointl[1] - a * pointl[0]
    y = int(a * x + b)
    return y


########################################################################
# drawLine function for the small slicing piece
########################################################################
def drawLine(contour, img):
    Y_list = getAllpoint(contour)
    for i in range(0, len(Y_list), 2):

        pointX = Y_list[i][0]
        pointY_top = Y_list[i][1]
        pointY_bottom = Y_list[i][2]

        # left line
        cv.line(img, (pointX, pointY_top), (pointX, pointY_bottom), (0, 0, 255), 1)


def suitForSlicing(list, midPoint):
    hasUpper = False
    hasLower = False
    for ele in list:
        if ele < midPoint:
            hasUpper = True
        else:
            hasLower = True
    if hasUpper is True and hasLower is True:
        return True
    else:
        return False


def findTop(list_V, midPoint):    # Y is the smallest
    if list_V[0] < list_V[-1]:
        if list_V[0] <= midPoint:
            return list_V[0]
        else:
            return -1
    else:
        if list_V[-1] <= midPoint:
            return list_V[-1]
        else:
            return -1


def findBottom(list_V, midPoint):    # Y is the largest
    if list_V[0] < list_V[-1]:
        if list_V[-1] >= midPoint:
            return list_V[-1]
        else:
            return -1
    else:
        if list_V[0] >= midPoint:
            return list_V[0]
        else:
            return -1


def findNextUpper(dict, index, mid):
    while True:
        if index in dict:
            vList = dict.get(index)
            if suitForSlicing(vList, mid):
                if vList[0] < vList[-1]:
                    return index, vList[0]
                else:
                    return index, vList[-1]
            else:
                if vList[0] <= mid:
                    return index, vList[0]
                else:
                    index += 1
        else:
            index += 1


def findNextLower(dict, index, mid):
    while True:
        if index in dict:
            vList = dict.get(index)
            if suitForSlicing(vList, mid):
                if vList[0] < vList[-1]:
                    return index, vList[-1]
                else:
                    return index, vList[0]
            else:
                if vList[0] >= mid:
                    return index, vList[0]
                else:
                    index += 1
        else:
            index += 1


def getAllpoint(contour):
    contour = sorted(contour, key=lambda tup: tup[0][0])  # sort the contour, from left point to right point

    # initial a dictionary
    contour_dict = {}  # create dictionary to record the contour point with the same X value, X: [Y1, Y2}...
    k = contour[0][0][0]
    v = []
    v.append(contour[0][0][1])
    contour_dict[k] = v
    contour = contour[1:]
    for ele in contour:
        key = ele[0][0]
        value = ele[0][1]
        if key in contour_dict:
            listValue = contour_dict.get(key)
            listValue.append(value)
        else:
            newValue = []
            newValue.append(value)
            contour_dict[key] = newValue

    # find the left, right point, and mid_y
    left_point, right_point = getLeftRight(contour_dict)
    y_mid = getMidPoint(left_point, right_point, left_point[0])
    startPoint_X = left_point[0]
    endPoint_X = right_point[0]

    ####
    # create a list, to record each pixel's Y-value, which used to slice. for example: (X, topY, bottomY)
    ####
    Y_list = []

    # add the first point (the most left) into list
    previousPoint_X = startPoint_X
    previousPoint_listValue = contour_dict.get(previousPoint_X)
    if len(previousPoint_listValue) > 1:
        previousTop = findTop(previousPoint_listValue, y_mid)
        previousBottom = findBottom(previousPoint_listValue, y_mid)
    else:
        previousTop = y_mid
        previousBottom = y_mid
    Y_list.append((startPoint_X, previousTop, previousBottom))

    # add others into list
    current_X = startPoint_X + 1
    while current_X <= endPoint_X:
        if current_X in contour_dict:
            current_listValue = contour_dict.get(current_X)  # list all y-values from all point with the same x-value
            current_mid = getMidPoint(left_point, right_point, current_X)
            topY = findTop(current_listValue, current_mid)
            bottomY = findBottom(current_listValue, current_mid)

            if not suitForSlicing(current_listValue, current_mid):  # or current_X == endPoint_X:
                if topY == bottomY:  # the most right point
                    print("right point")
                elif topY != -1:
                    bottomY = previousBottom
                elif bottomY != -1:
                    topY = previousTop
            Y_list.append((current_X, topY, bottomY))
            previousTop = topY
            previousBottom = bottomY

        else:
            Y_list.append((current_X, previousTop, previousBottom))
        current_X += 1

    # update the list
    size = len(Y_list)
    for i in range(0, size):
        k = Y_list[i][0]
        mid = getMidPoint(left_point, right_point, k)
        if k in contour_dict:
            vList = contour_dict.get(k)
            if suitForSlicing(vList, mid):
                continue
            else:
                originalTopY = findTop(vList, mid)
                originalBottomY = findBottom(vList, mid)

                if originalTopY != -1:
                    nextX, lower = findNextLower(contour_dict, k+1, mid)
                    temp = list(Y_list[i])
                    temp[2] = math.floor((lower - Y_list[i - 1][2]) / (nextX - k) + Y_list[i - 1][2])
                    temp = tuple(temp)
                    Y_list[i] = temp
                elif originalBottomY != -1:
                    nextX, upper = findNextUpper(contour_dict, k + 1, mid)
                    temp = list(Y_list[i])
                    temp[1] = math.floor((upper - Y_list[i - 1][1]) / (nextX - k) + Y_list[i - 1][1])
                    temp = tuple(temp)
                    Y_list[i] = temp
        else:
            nextX_u, upper = findNextUpper(contour_dict, k+1, mid)
            nextX_l, lower = findNextLower(contour_dict, k+1, mid)
            temp = list(Y_list[i])
            temp[1] = math.floor((upper - Y_list[i - 1][1]) / (nextX_u - k) + Y_list[i - 1][1])
            temp[2] = math.floor((lower - Y_list[i - 1][2]) / (nextX_l - k) + Y_list[i - 1][2])
            temp = tuple(temp)
            Y_list[i] = temp

    return Y_list


########################################################################
# get slicing rectangle box and return as np.array
########################################################################
def slicingRect(contour):
    Y_list = getAllpoint(contour)
    rect_array = []
    for i in range(0, len(Y_list)-1):
        a = (Y_list[i][1] + Y_list[i+1][1]) / 2
        b = (Y_list[i][2] + Y_list[i+1][2]) / 2
        rect_array.append((1, math.floor(b - a)))
    return rect_array


########################################################################
# calculate volume using slicing rectangle box
########################################################################
def getVolume(rectArr_side, rectArr_top, ratio, model):
    # scale factor length_side / length_top
    # V = pi * A * B * h

    if model == "ellip":
        # apply elliptic cylinder volume model here
        pi = 3.14159265
        A = np.array(rectArr_top)[:, 1] / ratio / 2
        B = np.array(rectArr_side)[:, 1] / ratio / 2
        h = np.array(rectArr_side)[:, 0] / ratio
        V = np.sum(pi * np.multiply(np.multiply(A, B), h))
    else:
        # apply rectangle volume model here
        A = np.array(rectArr_top)[:, 1] / ratio
        B = np.array(rectArr_side)[:, 1] / ratio
        h = np.array(rectArr_side)[:, 0] / ratio
        V = np.sum(np.multiply(np.multiply(A, B), h))
    return V


########################################################################
# Main View
########################################################################
def procView(HSVlower, HSVupper, img, ratio,  display, auto=False):

    if not auto:
        # convert the image to the hsv data format
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = cv.GaussianBlur(hsv, (3,3), 10)
        hsv = cv.dilate(hsv, (3, 3))
        # Threshold the HSV image to get only brown colors
        mask = cv.inRange(hsv, HSVlower, HSVupper)
        ret, thresh = cv.threshold(mask, 127, 255, 0)
        # ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    else:
        mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(thresh, 1, 2)
    contour = 0
    for cnt in contours:
        area = cv.contourArea(cnt)  # get the area of every contour, that is the area of the seed
        # filter all tiny area of contour
        if area < 2000:
            continue
        contour = cnt
    # minAreaRect() returns ( top-left corner(x,y), (width, height), angle of rotation )
    rect = cv.minAreaRect(contour)
    length = rect[1][0] / ratio
    width = rect[1][1] / ratio
    if length < width:
        width, length = length, width

    if display:
        drawLine(contour, img)
        cv.drawContours(img, [contour], 0, (255, 0, 0), 1)
    rectArray = slicingRect(contour)

    return length, width, rectArray


def newProcView(contour, img, ratio, display):

    rect = cv.minAreaRect(contour)
    length = math.floor(rect[1][0])
    width = math.floor(rect[1][1])
    length = length / ratio
    width = width / ratio
    if length < width:
        width, length = length, width

    if display:
        drawLine(contour, img)
        cv.drawContours(img, [contour], 0, (255, 0, 0), 1)
    rectArray = slicingRect(contour)

    return length, width, rectArray

########################################################################
# display the result
########################################################################
# print("The scale_factor is  %.2f"%scale_factor)
def displayResult(length_side, height, width, volume, img_side, img_top, img_side_saving, img_top_saving,
                  save, display):
    # construct the string show on the images

    str_length = "Length is  %.2f" % length_side + " mm"
    str_width = "Width is  %.2f" % width + " mm"
    str_height = "Thickness is  %.2f" % height + " mm"
    str_volume = "Volume is  %.2f" % volume + " mm^3"

    # show the result image with the source image for side view
    cv.putText(img_side, str_length, (10, 60), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img_side, str_height, (10, 90), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img_side, str_volume, (10, 120), 0, 1, (255, 255, 255), 2, cv.LINE_AA)

    # show the result image with the source image for top view
    cv.putText(img_top, str_length, (10, 60), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img_top, str_width, (10, 90), 0, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img_top, str_volume, (10, 120), 0, 1, (255, 255, 255), 2, cv.LINE_AA)

    if save:
        cv.imwrite(img_side_saving, img_side)
        cv.imwrite(img_top_saving, img_top)
    if display:
        cv.imshow('Side View', img_side)
        cv.imshow('Top View', img_top)
        cv.waitKey()

    return


########################################################################
# Calculate the volume using getVolume function
########################################################################
def procVolume(ratio, HSV_Upper, HSV_Lower_side, img_side, HSV_Lower_top, img_top,
               display, auto=False):

    contour_side, contour_top = normalizeImage(img_side, img_top, HSV_Lower_side, HSV_Lower_top, HSV_Upper)
    length_side, height, rectArray_side = newProcView(contour_side, img_side, ratio, display)
    length_top, width, rectArray_top = newProcView(contour_top, img_top, ratio, display)

    # # Calculate the length and width using procView function for side view
    # length_side, height, rectArray_side = procView(HSV_Lower_side, HSV_Upper, img_side, ratio, display, auto)
    #
    # # Calculate the length and width using procView function for top view
    # length_top, width, rectArray_top = procView(HSV_Lower_top, HSV_Upper, img_top, ratio, display, auto)

    return length_side, height, width, rectArray_side, rectArray_top


