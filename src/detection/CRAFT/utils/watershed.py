import cv2
import numpy as np
import math
import Polygon as plg

def watershed(oriimage, image, viz=False):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("binary", binary)
        cv2.waitKey()
    kernel = np.ones((3, 3), np.uint8)
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    sure_bg = mb
    if viz:
        cv2.imshow("sure_bg", mb)
        cv2.waitKey()
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)
    markers = labels.copy() + 1
    # markers = markers+1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()
    # a = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]

    if viz:
        color_markers = np.uint8(markers + 1)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers1", color_markers)
        cv2.waitKey()

    if viz:
        cv2.imshow("image", oriimage)
        cv2.waitKey()
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)
