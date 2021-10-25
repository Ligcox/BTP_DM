import cv2
import numpy as np
import time
import sys
import copy
import threading
import multiprocessing
import serial
import struct
import math
import random
import subprocess
import os

# 用于Trackbar回调
from functools import partial

from queue import Queue
import math
from config.config import *
from config.devConfig import *
from BCPloger import BCPloger

timeList = []
posList = []
velocityList = []
pointList = []
circleCenterPosList = []
TarData = []
Wise = [0]
cnt = [0]

loger = BCPloger()


# 基于角度(Degrees)的三角函数，大部分语言都使用弧度(Radians)作为标准
def sind(x):
    return np.sin(x * np.pi / 180)


def cosd(x):
    return np.cos(x * np.pi / 180)


# 针对opencv的矩形Rect的一些操作
# cv2 RotatedRect 数据格式
# [[x,y],[w,h],rotation]
# 顺带一提，最终画边框用drawContour的时候用的是BoxPoint，格式如下
# cv2 BoxPoints 数据格式
# [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

# 找到两个Rect长边朝向之间的夹角


def getCv2RotatedRectAngleDifference(rect1, rect2):
    return np.abs(getCv2RotatedRectOrientation(rect1) - getCv2RotatedRectOrientation(rect2))


# 找到两个Rect中心点的距离与较大Rect长边的比值
# TODO: 检测Rect中心点连线与较大Rect长边的夹角


def getCv2RotatedRectDistanceRatio(rect1, rect2):
    try:
        distance = np.sqrt(sum((np.array(rect1[0]) - np.array(rect2[0])) ** 2))
        rect1_length = max(rect1[1][0], rect1[1][1])
        rect2_length = max(rect2[1][0], rect2[1][1])
        val = distance * 1.0 / max(rect1_length, rect2_length)
    except Exception as e:
        print(e.args)
        # put ratio to maximum so that it will fail ratio test
        val = np.inf
    return val


# 找到Rect以较长边h时的倾角(也就是该Rect,aka 灯条 的朝向)


def getCv2RotatedRectOrientation(rect):
    val = rect[2]
    w, h = rect[1]
    if w > h:
        val += 90
    while val > 90:
        val -= 180
    while val <= -90:
        val += 180
    return val


def getCv2IsVerticalRect(rect):
    '''
    :breif: 检测灯条比例，刨除反光和血量条
            注意，在某些特定的血量下，血量条仍可能被误识别为装甲板灯条
    '''
    w, h = max(rect[1]), min(rect[1])
    # print(w,h)
    if w > 70:
        return False
    if 1.1 <= w / h <= 5:
        return True
    return False


# 图像处理辅助function，保证没有边被裁剪的同时完成缩放+旋转
def rotateImage(img, angle=0, scale=1):
    try:
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
        sa = np.abs(sind(angle))
        ca = np.abs(cosd(angle))
        outW = scale * (h * sa + w * ca)
        outH = scale * (h * ca + w * sa)
        M[0, 2] += (outW / 2) - cX
        M[1, 2] += (outH / 2) - cY
        return cv2.warpAffine(img, M, (int(outW), int(outH)))
    except Exception as e:
        print(e.args)
        return None


def boxPoints2ublr(box):
    '''
    :breif: 将两个灯条的boxpoint信息转换为上下左右的信息
    '''
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    # 上下左右四个点坐标
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

    if top_point_x >= bottom_point_x:
        vertices = np.array([
            [left_point_x, left_point_y],
            [bottom_point_x, bottom_point_y],
            [top_point_x, top_point_y],
            [right_point_x, right_point_y]
        ])
    else:
        vertices = np.array([
            [top_point_x, top_point_y],
            [left_point_x, left_point_y],
            [right_point_x, right_point_y],
            [bottom_point_x, bottom_point_y]
        ])
    return vertices


_t = []


def debug_show_window(winname, mtx):
    if DEBUG_MODEL:
        if winname not in _t:
            _t.append(winname)
        cv2.imshow(winname, mtx)


def SortCoordinate(boxes: list) -> list:
    """
    坐标排序 顺序为 左上 右上 左下 右下
    """
    l = len(boxes)
    for i in range(l):
        j = i
        for j in range(l):
            if (boxes[i][0] > boxes[j][0]):
                boxes[i], boxes[j] = boxes[j], boxes[i]
            if (boxes[i][1] <= boxes[j][1]):
                boxes[i], boxes[j] = boxes[j], boxes[i]
    if boxes[0][0] > boxes[1][0]:
        boxes[0], boxes[1] = boxes[1], boxes[0]
    if boxes[2][0] > boxes[3][0]:
        boxes[2], boxes[3] = boxes[3], boxes[2]
    return boxes


def ExpandRegion(boxes: list, ExpandSize: tuple):
    """
    区域增大
    """
    boxes = SortCoordinate(boxes)
    w, h = ExpandSize
    for k in range(len(boxes)):
        if k == 0:
            boxes[k] = list(np.add(np.array(boxes[k]), np.array([w, -h])))
        if k == 1:
            boxes[k] = list(np.add(np.array(boxes[k]), np.array([-w, -h])))
        if k == 2:
            boxes[k] = list(np.add(np.array(boxes[k]), np.array([w, h])))
        if k == 3:
            boxes[k] = list(np.add(np.array(boxes[k]), np.array([-w, h])))
    return boxes


def TransformPerspective(boxes, size: tuple):
    """
    透视变换
    """
    width, height = size
    pts1 = np.float32(boxes)  # 截取对片中的哪个区域
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # 定义显示的卡片的坐标
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 两个区域坐标绑定
    return matrix


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""

    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src, alpha):
    """图像亮度增强"""

    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    # alpha = int(get_lightness(src) * 0.085)
    # if alpha < 0:
    #     alpha = 0
    max_percentile_pixel, min_percentile_pixel = compute(src, 1 + alpha, 99 - alpha)
    # max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.99, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    # print(lightness)
    return lightness


def calLineRadians(point1: tuple, point2: tuple) -> float:
    """
    计算直线与坐标轴夹角大小（弧度制）
    :param point1: 点1
    :param point2: 点2
    :return: 弧度
    """
    h = point1[1] - point2[1]
    w = point1[0] - point2[0]
    radians = math.atan2(h, w)

    return radians


def calDisplacement(pos: tuple):
    """
    计算位移
    :return: 位移差 pixel
    """
    posList.append(pos)

    Len = len(posList)
    if Len >= 2:
        posArray = np.array(posList)
        xList = list(posArray[:, 0])
        yList = list(posArray[:, 1])
        xVar = calVariance(xList)
        yVar = calVariance(yList)
        print(xVar, yVar)
        if xVar <= 0.5 and yVar <= 0.5:
            Dis = 0

        else:
            XDis = posList[Len - 1][0] - posList[Len - 2][0]
            YDis = posList[Len - 1][1] - posList[Len - 2][1]
            Dis = math.sqrt(XDis ** 2 + YDis ** 2)
        if Len == 10:
            posList.remove(posList[0])
    elif Len < 2:
        Dis = 0
    return Dis


def calCircleCenterDisplacement(pos):
    circleCenterPosList.append(pos)
    XDis = circleCenterPosList[0][0] - pos[0]
    YDis = circleCenterPosList[0][1] - pos[1]
    Dis = math.sqrt(XDis ** 2 + YDis ** 2)
    if len(circleCenterPosList) == 2:
        circleCenterPosList.remove(circleCenterPosList[0])
    return Dis


def calVariance(List):
    variance = np.var(List)
    return variance


def calPointDistance(point1: tuple, point2: tuple) -> float:
    # ***** 求两点间距离***** #
    distance = math.pow((point1[0] - point2[0]), 2) + math.pow((point1[1] - point2[1]), 2)
    distance = math.sqrt(distance)
    return distance


def calTimeDiffer():
    """
    计算时间差
    :return: 时间差 s
    """
    Time = time.time()
    timeList.append(Time)
    if len(timeList) == 2:
        T = timeList[1] - timeList[0]
        timeList.remove(timeList[0])
        return T
    else:
        return 0


def calAvgVelocity(velocity: float, clear: bool = False) -> float:
    """
    计算速度平均值
    :param velocity: 线速度
    :param clear: 是否清空速度列表
    :return: 速度平均值
    """
    Avg = 0
    if clear:
        if len(velocityList) != 0:
            velocityArray = np.array(velocityList)
            Avg = np.mean(velocityArray)
        else:
            velocityList.clear()
        return Avg
    else:
        velocityList.append(velocity)
        if len(velocityList) == 20:
            velocityList.remove(velocityList[0])
        velocityArray = np.array(velocityList)
        Avg = np.mean(velocityArray)
        return Avg


def calAvgPoint(inputPoint):
    avgPoint = ()
    pointList.append(inputPoint)
    if len(pointList) == 20:
        pointList.remove(pointList[0])
    pointArray = np.array(pointList)
    avgPoint = np.mean(pointArray, axis=0).tolist()
    dis = calPointDistance(avgPoint, inputPoint)
    if dis > 20:
        pointList.clear()
    return avgPoint




def calLengthWidthRatio(box):
    if len(box) > 0:
        height = calPointDistance(box[0], box[2])  # 轮廓长度
        weight = calPointDistance(box[0], box[1])  # 轮廓宽度

        if height > weight:
            height, weight = weight, height
        if height != 0:
            ratio = float(weight) / float(height)  # 长宽比
        if height == 0:
            ratio = None
    else:
        ratio = None
    return ratio


def calCenter(box: list) -> tuple:
    """
    计算矩形中心点
    :param box: 点坐标信息
    :return: 中心点
    """
    center = (int(box[0][0] + (box[2][0] - box[0][0]) / 2), int(box[0][1] + (box[2][1] - box[0][1]) / 2))
    return center


def calMomentCenter(ContoursList):
    momentCenterPoint = ()
    if len(ContoursList) != 0:
        contour = max(ContoursList, key=cv2.contourArea)

        mm = cv2.moments(contour)  # 求轮廓的几何矩
        cx = mm['m10'] / mm['m00']  # 原点的零阶矩
        cy = mm['m01'] / mm['m00']
        momentCenterPoint = (int(cx), int(cy))  # 几何矩中心
    return momentCenterPoint


def recordTarData(velocity, degree, radius, x, y):
    """
    速度 与x轴夹角 旋转半径
    :return:
    """
    TarData.append([velocity, degree, radius, x, y])
    if len(TarData) == 100:
        TarData.remove(TarData[0])


def getTarData(num):
    if len(TarData) != 0:
        dataArray = np.array(TarData)[:, num]
        return dataArray
    else:
        return []


def setLastWise(wise):
    Wise[0] = wise


def getLastWise():
    return Wise[0]


def counter(zero: False):
    if zero is True:
        cnt[0] = 0
    else:
        cnt[0] += 1
    return cnt[0]


def abs_max_filter(val, maxval):
    if abs(val) >= abs(maxval):
        sign = 1 if val > 0 else -1
        res = sign * abs(maxval)
    else:
        res = val
    return res


def abs_min_filter(val, minval):
    if abs(val) <= abs(minval):
        res = 0
    else:
        res = val
    return res
