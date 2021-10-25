# -*- coding: UTF-8 -*-
"""
TargetPredictor 目标预测层
目标预测层简单介绍 通过TargetDetector识别到的目标来进行预测
开发者可以在这层对识别到的目标进行预测，可使用卡拉曼滤波来实现。
"""
import time
from math import *
from module import *
from utils import *
from config.config import *
import numpy as np
from numpy.linalg import inv
from BCPloger import *
from targetDetector import Target

loger = BCPloger()


class Predictor(module):
    name = "Empty Predictor Control"

    def process(self, frame, coordinate):
        """
        :brief:预测处理
        :param frame: 原图像
        :return: None
        """
        pass

    def drawRoi(self, image, rect: None, point=None, color=(255, 0, 255), thickness=2):
        if rect is not None:
            cv2.drawContours(image, [rect], -1, color, 2)

        if point is not None:
            point_array = np.array(point, dtype='int16')
            cv2.circle(image, tuple(point_array), 3, color, -1)

        return image

    def updateProcess(self, frame, rects=None, points=None):
        img = frame
        if self.checkData(rects):
            rect = rects
            img = self.drawRoi(img.copy(), rect, point=None)

        if self.checkData(points):
            # shape = np.shape(points)

            point = points
            img = self.drawRoi(img.copy(), rect=None, point=point)

        debug_show_window("predictImg", img)
        return img

    def checkData(self, data) -> bool:
        if data is not None and len(data) != 0:
            return True
        else:
            return False

    def convertDataFormat(self, rect, cvtRectToDataList=True, cvtRectToBoxPoints=True):
        dataList = []
        box = []
        if len(rect) != 0:
            if cvtRectToDataList is True:
                data = [None for i in range(ROI_DATA_LENGTH)]
                data[ROI_RECT] = rect

                data[ROI_BOX] = np.int0(cv2.boxPoints(data[ROI_RECT]))

                data[PNP_INFO] = self.pnp_info(data[ROI_BOX])

                if data[PNP_INFO] is not None:
                    dataList.append(data)
            elif cvtRectToBoxPoints is True:
                box = cv2.boxPoints(rect)  # 转换为long类型
                box = np.int0(box)
        if cvtRectToDataList is True:
            return dataList
        elif cvtRectToBoxPoints is True:
            return box
        else:
            return []


class KalmanFilter(object):
    def __init__(self, args):
        self.A = args["A"]
        self.B = args["B"]
        self.H = args["H"]
        self.Q = args["Q"]
        self.R = args["R"]
        initial_x = args["initial_x"]
        initial_p = args["initial_p"]

        self.pre_best_estimate_x = initial_x
        self.cur_estimate_x = None
        self.cur_best_estimate_x = None

        self.pre_best_estimate_p = initial_p
        self.cur_estimate_p = None
        self.cur_best_estimate_p = None

    def process(self, measure_state, control, down_time):
        self.control = control
        self.prediction(down_time)
        self.update(measure_state, control)

    def update(self, measure_state, control):
        self.S = np.matmul(self.H, np.matmul(self.cur_estimate_p, self.H.T)) + self.R
        self.K = np.matmul(self.cur_estimate_p, np.matmul(self.H.T, inv(self.S)))
        self.y = measure_state - np.dot(self.H, self.cur_estimate_x)

        self.cur_best_estimate_x = self.cur_estimate_x + np.dot(self.K, self.y)
        self.cur_best_estimate_p = np.matmul(np.eye(4) - np.matmul(self.K, self.H), self.cur_estimate_p)

        self.pre_best_estimate_x = self.cur_best_estimate_x
        self.pre_best_estimate_p = self.cur_best_estimate_p

    def prediction(self, down_time):
        self.A = np.array([
            [1, 0, down_time, 0],
            [0, 1, 0, down_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.cur_estimate_x = np.dot(self.A, self.pre_best_estimate_x) + np.dot(self.B, self.control)
        self.cur_estimate_p = np.matmul(self.A, np.matmul(self.pre_best_estimate_p, self.A.T)) + self.Q


class EnergyPredictor(Predictor):
    def __init__(self, hide_controls=False):
        self.controls = energy_predictor_controls
        self.name = "energy_predictor"
        super().__init__(hide_controls)

    def process(self, frame, dataList):
        wise = self.judgeDirectionRotation()
        mode = "lead_angle"
        predictTargetCenter, predictRect = self.predictTarget(wise, dataList)
        TargetList = self.convertDataFormat(predictRect)

        self.updateProcess(frame, predictRect, (predictTargetCenter))

        return TargetList

    def judgeMode(self):
        pass

    def judgeDirectionRotation(self):
        data_array = getTarData(1)
        if len(data_array) != 0:
            length = len(data_array)
            if length > 10:

                degree = data_array[length - 10:length - 1]  # 数据截取

                degree[degree < 0] += 360  # 将负角度统一加360

                if max(degree) - min(degree) > 350:
                    return getLastWise()
                #  逆时针
                if 15 >= degree[0] - degree[8] >= 0:
                    setLastWise(anticlockwise)
                    return anticlockwise
                #  顺时针
                elif -15 < degree[0] - degree[8] < 0:
                    setLastWise(clockwise)
                    return clockwise
                else:
                    return getLastWise()
            else:
                return getLastWise()
        else:
            return getLastWise()

    def predictTarget(self, wise, data_list):
        origin_radian, circle_center = data_list[0], data_list[1]
        energy_target_center, energy_target_rect = data_list[2]
        predict_target_center = self.predictTarPoint(wise, circle_center, energy_target_center,
                                                     origin_radian)  # 预测击打点
        predict_rect = self.predictTarRect(predict_target_center, energy_target_rect)  # 预测击打点的矩形框
        return predict_target_center, predict_rect

    def predictTarPoint(self, wise, rotating_circle_center, tar_center_point, origin_radian):
        predict_point = ()
        if len(rotating_circle_center) != 0 and len(tar_center_point) != 0:
            x, y = rotating_circle_center
            displacement = calDisplacement(tar_center_point)  # 计算上一帧与这一帧目标中心点位移
            circle_center_displacement = calCircleCenterDisplacement(rotating_circle_center)
            radius = calPointDistance(rotating_circle_center, tar_center_point)  # 目标中心与旋转中心的距离
            if displacement > 0:
                time_differ = calTimeDiffer()  # 计算两帧的时间差
                if time_differ != 0:
                    Velocity = displacement / time_differ  # 计算目标中心点移动速度
                    # if displacement > 60 or circle_center_displacement > 20:
                    #     Velocity = calAvgVelocity(Velocity, True)
                    # else:
                    #     Velocity = calAvgVelocity(Velocity)
                    recordTarData(Velocity, math.degrees(origin_radian), radius, rotating_circle_center[0],
                                  rotating_circle_center[1])
                    # if mode == "lead_time":
                    #     leadTime = self.getControlVal(mode)
                    #     S = Velocity * leadTime
                    #     predictAngle = 180 * S / (math.pi * radius)
                    #     if predictAngle > 60:
                    #         predictAngle = 60
                    # elif mode == "lead_angle":
                    #     predictAngle = self.getControlVal("lead_angle")

                    predict_angle = self.getControlVal("lead_angle")
                    predict_radian = math.radians(predict_angle)
                    if wise == anticlockwise:
                        final_radian = origin_radian - predict_radian
                    elif wise == clockwise:
                        final_radian = origin_radian + predict_radian
                    elif wise is None:
                        final_radian = origin_radian
                    predictX = int(x - radius * math.cos(final_radian))
                    predictY = int(y - radius * math.sin(final_radian))

                    predictPoint = (predictX, predictY)
            return predictPoint

    def predictTarRect(self, center, energyTarget):
        TarRect = []
        if energyTarget is not None and center is not None:
            if len(energyTarget) != 0 and len(center) != 0:
                rect = energyTarget
                weight = rect[1][0]
                height = rect[1][1]
                angel = rect[2]
                TarRect = (center, (weight, height), angel)
        return TarRect

    def pnp_info(self, armor_box):
        '''
        breif:pnp解算结果
        lightstrip1, lightstrip2:两个灯条的boxPoint信息
        return: [距离， yaw偏转角，pitch偏转角]
        '''
        armour_type = self.getControlVal("energyArmour")
        w, h = armour_type
        w, h = w / 2, h / 2
        world_coordinate = np.array([
            [w, h, 0],
            [-w, h, 0],
            [-w, -h, 0],
            [w, -h, 0],

        ], dtype=np.float64)

        # 像素坐标
        pnts = np.array(armor_box, dtype=np.float64)

        # rotation_vector 旋转向量 translation_vector 平移向量
        success, rvec, tvec = cv2.solvePnP(world_coordinate, pnts, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])
        distance = np.linalg.norm(tvec)

        yaw_angle = np.arctan(tvec[0] / tvec[2])
        pitch_angle = np.arctan(tvec[1] / tvec[2])
        # 默认为弧度制，转换为角度制改下面
        # 这里角度为像素坐标系值，图像中右侧为x正方向，下侧为y轴正方向
        yaw_angle = float(np.rad2deg(yaw_angle))
        pitch_angle = -float(np.rad2deg(pitch_angle))

        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, roll, yaw = eulerAngles[0], eulerAngles[1], eulerAngles[2]
        # print(pitch, yaw, roll)
        pnp_list = [distance, yaw_angle, pitch_angle]

        # print(f"{distance},{yaw_angle},{pitch_angle}")

        # 根据pnp解算结果做第一次筛选
        # 直接筛选出有问题的目标

        return pnp_list

    def convertDataFormat(self, rect, cvtRectToDataList=True, cvtRectToBoxPoints=True):
        dataList = []
        box = []
        if len(rect) != 0:
            if cvtRectToDataList is True:
                data = [None for i in range(ROI_DATA_LENGTH)]
                data[ROI_RECT] = rect

                data[ROI_BOX] = np.int0(cv2.boxPoints(data[ROI_RECT]))

                data[PNP_INFO] = self.pnp_info(data[ROI_BOX])

                if data[PNP_INFO] is not None:
                    dataList.append(data)
            elif cvtRectToBoxPoints is True:
                box = cv2.boxPoints(rect)  # 转换为long类型
                box = np.int0(box)
        if cvtRectToDataList is True:
            return dataList
        elif cvtRectToBoxPoints is True:
            return box
        else:
            return []

    def updateProcess(self, frame, rect=None, point=None):
        img = frame
        if rect is not None:
            if len(rect) != 0:
                Point = rect[0]
                img = self.drawRoi(img.copy(), rect, point=Point)
        if point is not None:
            if point != ():
                img = self.drawRoi(img.copy(), rect=None, point=point)

        debug_show_window("predictImg", img)
        return img

    def drawRoi(self, image, rect: None, point=None, color=(255, 0, 255), thickness=2):

        if rect is not None:
            boxPoints = self.convertDataFormat(rect, cvtRectToDataList=False, cvtRectToBoxPoints=True)
            cv2.drawContours(image, [boxPoints], -1, color, 2)

        if point is not None:
            pointArray = np.array(point, dtype='int16')

            cv2.circle(image, tuple(pointArray), 3, color, -1)

        return image


class ArmourPredictor(Predictor):
    def __init__(self, hide_controls=True):
        self.controls = predict_controls
        self.name = "KalmanFilter"
        self.kf_model = KalmanFilter(predict_controls)
        self.control = np.array([0, 0, 0, 0])
        self.pre_X, self.pre_Y = 0, 0
        self.advance_list = [0 for i in range(10)]
        super().__init__(hide_controls)

    def process(self, frame, rect_info: Target):
        if rect_info.target_exist == True:
            down_time = self.calAdvance(rect_info.rect_pnp_info)
            X, Y = self.kalmanPredict(rect_info.rect_rotate_info[0], down_time)
            rect_rotate_info = rect_info.rect_rotate_info
            predict_rect_rotate_info = ((X, Y), rect_rotate_info[1], rect_rotate_info[2])
            box = np.int0(cv2.boxPoints(predict_rect_rotate_info))  # 转换为long类型
            self.updateProcess(frame, rects=box, points=None)
            return X, Y

    def kalmanPredict(self, coordinate, down_time):
        X, Y = int(coordinate[0]), int(coordinate[1])
        cur_state = np.array([X, Y, (X - self.pre_X) / 0.0366, (Y - self.pre_Y) / 0.0366])

        # Gaussian Noise
        measure_noise = np.random.multivariate_normal([0, 0, 0, 0], self.controls["N"])
        # Apply measurement, z_k = H_k * x_k + V_k
        measure_state = np.dot(self.controls["H"], cur_state) + measure_noise

        self.kf_model.process(measure_state, self.control, down_time)

        self.pre_X, self.pre_Y = X, Y
        return self.kf_model.cur_best_estimate_x[0], self.kf_model.cur_best_estimate_x[1]

    def calAdvance(self, pnp_info):
        ball_speed = 15
        distance = int(pnp_info[0])
        ration = radians(pnp_info[2])
        Y = distance * sin(ration)
        down_time = abs(Y / (1000.0 * ball_speed))  # 下坠时间
        fly_time = distance / (800 * ball_speed)
        advance = fly_time + down_time
        self.advance_list.remove(self.advance_list[0])
        self.advance_list.append(advance)
        advance = self.np_move_avg(np.array(self.advance_list), 3)[9]
        return advance

    def np_move_avg(self, a, n, mode="same"):
        return (np.convolve(a, np.ones(n) / n, mode=mode))
