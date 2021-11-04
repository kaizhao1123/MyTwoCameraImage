import cv2 as cv
import numpy as np
from volume import procVolume, displayResult, getVolume
import os
import sys
import time
import pandas as pd


def main():

    # initial setup
    directory = 'pic/55'
    vint_side = 142
    vint_top = 142
    ratio = 95/3.945  # 23.954372623574  # 36.9496855346

    save = True
    display = True
    model = "ellip"  # rect

    # read images
    sidePath = directory + '/' + 'side.bmp'
    img_side = cv.imread(sidePath)
    topPath = directory + '/' + 'top.bmp'
    img_top = cv.imread(topPath)

    # set images name for saving
    img_side_saving = (directory + '/' + 'side-slice-' + '.jpg')
    img_top_saving = (directory + '/' + 'top-slice-' + '.jpg')

    # set the HSV range for side view and top view for wheat
    HSV_Lower_side = np.array([0, 0, vint_side])
    HSV_Lower_top = np.array([0, 0, vint_top])
    HSV_Upper = np.array([255, 255, 255])

    length_side, height, width, rectArray_side, rectArray_top = procVolume(ratio, HSV_Upper,
                                                                                        HSV_Lower_side, img_side,
                                                                                        HSV_Lower_top, img_top,
                                                                                        display)
    volume = getVolume(rectArray_side, rectArray_top, ratio, model)
    print("*************")
    print("volume=%.4f, length=%.4f, width=%.4f, height=%.4f" % (volume, length_side, width, height))
    print("*************")

    if display:
        displayResult(length_side, height, width, volume, img_side, img_top, img_side_saving,
                      img_top_saving,
                      save, display)
    return


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total time: --- %s seconds ---" % (time.time() - start_time) + "\n")
