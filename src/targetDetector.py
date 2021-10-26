import math
import time

import cv2
import numpy as np

from module import *
from utils import *
from config.config import *

Camera_intrinsic = {
    "mtx": np.array([[1.32280384e+03, 0.00000000e+00, 3.09015137e+02],
                     [0.00000000e+00, 1.32691122e+03, 2.04586526e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                    dtype=np.double),
    "dist": np.array([2.50140551e-03, -3.05940539e+00, -9.93333169e-03,
                      -3.64458084e-04, 2.33302573e+01], dtype=np.double),
}


class Target:
    def __init__(self, name: str, rect_info=[], armour_size=[], threshold_interval=0.08):
        self.name = name
        self.rect_info = [rect_info, None, None]
        self.armour_size = armour_size
        self.target_exist = False
        self._threshold_interval = threshold_interval
        self._timer = time.time()

    def checkTarget(self, latest_rect_info):
        if len(self.rect_info[ROI_RECT]) == 0:
            self.updateRectInfo(latest_rect_info)
            self.updateBoxState()
            update_obj = False
        elif self.checkInterval():
            if self.checkIOU(latest_rect_info):
                self.updateRectInfo(latest_rect_info)
                self.updateBoxState()
                update_obj = False
            else:
                update_obj = True
        else:
            update_obj = True
        return update_obj

    def updateRectInfo(self, latest_box_info):
        self.rect_info[ROI_RECT] = latest_box_info  # [(x,y),(w,h),angle]
        self.rect_info[ROI_BOX] = self.cvtToBoxPoint(latest_box_info)  # [(x1,y1),(x2,y2),(x3,y3)，(x4,y4)]
        self.rect_info[PNP_INFO] = self.cvtToBoxPnp(self.rect_info[ROI_BOX])  # [distance, int(yaw_angle), int(pitch_angle)]


    def updateBoxState(self):
        self.target_exist = True
        self._timer = time.time()

    def checkInterval(self):
        latest_time = time.time()
        interval = latest_time - self._timer
        if interval <= self._threshold_interval:
            return True
        else:
            return False

    def checkIOU(self, latest_box_info, threshold=0.5):
        """检查图像框交并补"""
        original_box = self.rect_info[ROI_RECT]
        int_pts = cv2.rotatedRectangleIntersection(original_box, latest_box_info)[1]
        if int_pts is not None:
            area1, area2 = original_box[1][0] * original_box[1][1], latest_box_info[1][0] * latest_box_info[1][1]
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            int_area = cv2.contourArea(order_pts)
            iou = int_area / (area1 + area2 - int_area)
        else:
            iou = 0
        if iou >= threshold:
            return True
        elif iou < threshold:
            return False

    def cvtToBoxPoint(self,box_info):
        """数据格式转换，转换至目标角点信息"""
        rect_point_info = np.int0(cv2.boxPoints(box_info))  # 转换为long类型
        return rect_point_info

    def cvtToBoxPnp(self, box_info):
        """数据格式转换，转换至目标位姿信息"""
        w, h = self.armour_size
        w, h = w / 2, h / 2
        world_coordinate = np.array([
            [w, h, 0],
            [-w, h, 0],
            [-w, -h, 0],
            [w, -h, 0],

        ], dtype=np.float64)
        # 像素坐标
        pnts = np.array(box_info, dtype=np.float64)
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
        rect_pnp_info = [distance, int(yaw_angle), int(pitch_angle)]
        return rect_pnp_info
        # print(f"{distance},{yaw_angle},{pitch_angle}")


class targetDetector(module):
    name = "Empty Detector Control"

    def process(self, frame):
        return None


class simpleDetector(targetDetector):
    def __init__(self, hide_controls=True):
        self.controls = simple_detector_controls
        self.name = "SimpleDetector"
        super().__init__(hide_controls)

    def detectLightStrip(self, image=None):
        '''
        breif:返回灯条数据
        return:strip_list
        strip_list:为图像中全部装甲板灯条
                   0>>ROI_RECT：装甲板长方形框信息，输出结果为cv2.minAreaRect ((x, y), (w, h), θ )
                   1>>ROI_BOX:装甲板长方形框信息，输出结果为cv2.boxPoints((x0, y0), (x1, y1))
        '''
        try:
            if image.shape[2] == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                gray_img = image
            else:
                print('Image Shape {} Unexpected'.format(image.shape))
                return []
        except Exception as e:
            print(e.args)
            return []
        blur_img = cv2.blur(gray_img, (9, 9))
        _, thresh_img = cv2.threshold(blur_img, 2, 255, 0)
        blur_thresh = cv2.blur(thresh_img, (3, 3))
        contours, _ = cv2.findContours(
            blur_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        strip_list = []
        contour_area_threshold = image.shape[0] * image.shape[1] * \
                                 self.getControlVal('Contour_Area_Threshold')
        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area > contour_area_threshold:
                strip_data = [None for i in range(ROI_DATA_LENGTH)]
                strip_data[ROI_RECT] = cv2.minAreaRect(c)
                strip_data[ROI_BOX] = np.int0(
                    cv2.boxPoints(strip_data[ROI_RECT]))
                isVertical = getCv2IsVerticalRect(strip_data[ROI_RECT])
                if not isVertical:
                    continue
                strip_list.append(strip_data)
        return strip_list

    def getLightStripRelation(self, strip1, strip2):
        return [
            getCv2RotatedRectAngleDifference(
                strip1[ROI_RECT], strip2[ROI_RECT]),
            getCv2RotatedRectDistanceRatio(
                strip1[ROI_RECT], strip2[ROI_RECT])
        ]

    def detectArmor(self, lightstrip_list=[]):
        '''
        breif:返回装甲板数据
        return:armor_list
        armor_list:为图像中全部装甲板信息长度为ROI_DATALENTH
                   ROI_RECT：装甲板长方形框信息，输出结果为cv2.minAreaRect ((x, y), (w, h), θ )
                   ROI_BOX:装甲板长方形框信息，输出结果为cv2.boxPoints((x0, y0), (x1, y1))
                   PNP_INFO:装甲板pnp解算数据，输出结果为[距离， yaw偏转角，pitch偏转角]
        '''
        min_distance_ratio = {
            # 灯条编号: [匹配装甲板编号, 最小比例]
        }
        min_distance_ratio_data = {
            # 灯条编号: data
        }
        for i in range(len(lightstrip_list)):
            for j in range(i + 1, len(lightstrip_list)):
                angle_differnce, distance_ratio = self.getLightStripRelation(
                    lightstrip_list[i], lightstrip_list[j])
                # print(angle_differnce, distance_ratio)
                if (
                        angle_differnce <= self.getControlVal(
                    'LightStrip_Angle_Diff')
                        and self.getControlVal(
                    'LightStrip_Displacement_Diff_Min') <= distance_ratio <= self.getControlVal(
                    'LightStrip_Displacement_Diff_Max')):
                    min_distance_ratio[i] = min_distance_ratio.get(i, [j, 0xffff])
                    min_distance_ratio[j] = min_distance_ratio.get(j, [i, 0xffff])
                    if distance_ratio < min_distance_ratio[i][1] and distance_ratio < min_distance_ratio[j][1]:
                        armor_data = [None for i in range(ROI_DATA_LENGTH)]
                        armor_data[ROI_RECT] = cv2.minAreaRect(np.concatenate(
                            (lightstrip_list[i][ROI_BOX], lightstrip_list[j][ROI_BOX])))
                        armor_data[ROI_BOX] = np.int0(
                            cv2.boxPoints(armor_data[ROI_RECT]))
                        armor_data[PNP_INFO] = self.pnp_info(
                            lightstrip_list[i][ROI_BOX], lightstrip_list[j][ROI_BOX], armor_data[ROI_RECT])
                        if armor_data[PNP_INFO] is not None:
                            # 赋值新的装甲板数据
                            min_distance_ratio_data[i] = armor_data
                            min_distance_ratio_data[j] = armor_data
                            min_distance_ratio[i] = [j, distance_ratio]
                            min_distance_ratio[j] = [i, distance_ratio]
                        else:
                            continue
        armor_list = []
        for i in min_distance_ratio_data:
            if not min_distance_ratio_data[i] in armor_list:
                armor_list.append(min_distance_ratio_data[i])
        return armor_list

    def pnp_info(self, lightstrip1, lightstrip2, boxes):
        '''
        breif:pnp解算结果
        lightstrip1, lightstrip2:两个灯条的boxPoint信息
        return: [距离， yaw偏转角，pitch偏转角]
        '''
        # armour_type = self.getControlVal("bigArmour")
        lightstrip1 = boxPoints2ublr(lightstrip1)
        lightstrip2 = boxPoints2ublr(lightstrip2)

        if lightstrip1[0][0] > lightstrip2[0][0]:
            lightstrip1, lightstrip2 = lightstrip2, lightstrip1

        rect_w, rect_h = max(boxes[1]), min(boxes[1])

        # print(rect_w / rect_h)
        if rect_w / rect_h <= 3:
            armour_type = self.getControlVal("normalArmour")
        elif rect_w / rect_h >= 3:
            return None
        else:
            armour_type = self.getControlVal("bigArmour")

        # print(armour_type)
        w, h = armour_type
        w, h = w / 2, h / 2
        lightstrip_half_size = 5
        world_coordinate = np.array([
            [-w - lightstrip_half_size, -h, 0],
            [-w - lightstrip_half_size, -h, 0],
            [-w + lightstrip_half_size, h, 0],
            [-w + lightstrip_half_size, h, 0],
            [w - lightstrip_half_size, -h, 0],
            [w - lightstrip_half_size, -h, 0],
            [w + lightstrip_half_size, h, 0],
            [w + lightstrip_half_size, h, 0]
        ], dtype=np.float64)

        # 像素坐标
        pnts = np.array(np.vstack((lightstrip1, lightstrip2)),
                        dtype=np.float64)

        # rotation_vector 旋转向量 translation_vector 平移向量
        success, rvec, tvec = cv2.solvePnP(
            world_coordinate, pnts, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])
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
        if pnp_list[0] > 10:
            return pnp_list
        else:
            return None

    def createRoiMask(self, image, roi_list):
        mask = np.zeros(image, shape[:2], np.uint8) * 255
        boxes = []
        for r in roi_list:
            boxes.append(r[ROI_BOX])
        cv2.drawContours(image, boxes, -1, 255, -1)
        return mask

    def drawRoi(self, image, roi_list, color=(255, 0, 255), thickness=10):
        boxes = []
        center_points = []
        for r in roi_list:
            boxes.append(r[ROI_BOX])
        for r in roi_list:
            cv2.circle(image, tuple(int(i)
                                    for i in r[ROI_RECT][0]), 0, color, thickness)
        cv2.drawContours(image, boxes, -1, color, thickness)
        return image

    def process(self, image):
        lightstrips = self.detectLightStrip(image)
        armors = self.detectArmor(lightstrips)
        self.updateProcess(image, lightstrips, armors)
        return armors, len(lightstrips)

    def updateProcess(self, frame, light_strips=[], armors=[]):
        if not (self.getControlVal('silent') or (frame is None)):
            img = frame.copy()
            cv2.putText(img, 'ROI', (0, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            img0 = self.drawRoi(img, light_strips, self.getControlVal(
                'LightStrip_Box_Color'), self.getControlVal('LightStrip_Box_Thickness'))
            img = self.drawRoi(img0, armors, self.getControlVal(
                'Armor_Box_Color'), self.getControlVal('Armor_Box_Thickness'))
        else:
            img = BLANK_IMG
        debug_show_window(self.control_window_name + "img", img)
        cv2.waitKey(1)


class EnergyDetector(module):

    def __init__(self, hide_controls=False):
        self.controls = energy_detector_controls
        self.name = "energy_detector"
        self.target = Target("energyArmour", armour_size=[60, 70])
        super().__init__(hide_controls)

    def process(self, processed_img, origin_img):
        """
        寻找能量机关目标
        :return:TargetList（ROI类型）, dataList
        """
        contours = self.enhanceContour(processed_img.copy())  # 图像轮廓加强

        strip_list, strip_center_list = self.findStrip(contours)  # 能量机关寻找灯条和灯条中心

        # 灯条存在
        # if len(strip_list[Energy_Strip]) != 0:
            # energy_target_center, energy_target_rect = self.detectTarget(strip_list[Energy_Strip],
            #                                                              processed_img.copy())  # 灯条筛选

            # 估计旋转圆圆心，计算能量机关目标当前旋转角度
            # assumption_circle_center, origin_radian = self.calRotatingCircleCenter(
            #     strip_center_list[Energy_Strip_Point], energy_target_center)
            #
            # # 数据合并进列表
            # circle_center = strip_center_list[Circle_Center_Point]
            # energy_target = [energy_target_center, energy_target_rect]
            # data_list = [origin_radian, circle_center, energy_target]
            # 图像绘制
        #     if len(energy_target_rect) > 0:
        #         new_obj = self.target.checkTarget(energy_target_rect)
        #         if new_obj is True:
        #             self.target = Target("energyArmour", energy_target_rect, armour_size=[230, 127])
        #     elif len(energy_target_rect) == 0:
        #         self.target = Target("energyArmour", armour_size=[230, 127])
        #     self.updateProcess(origin_img, self.target.rect_point_info, (circle_center, energy_target_center))
        # return self.target

        target_rect = self.detectTarget(strip_list[Circle_Strip])
        if len(strip_list[Circle_Strip]) > 0:
            new_obj = self.target.checkTarget(target_rect)
            if new_obj is True:
                self.target = Target("energyArmour", target_rect, armour_size=[60, 70])

            self.updateProcess(origin_img, self.target.rect_info[ROI_BOX],
                               (strip_center_list[Circle_Center_Point], strip_center_list[Circle_Center_Point]))
        elif len(strip_list[Circle_Strip]) == 0:
            self.target = Target("energyArmour", armour_size=[60, 70])

        # 灯条存在
        if len(strip_list[Energy_Strip]) != 0:
            pass
        return self.target

    def enhanceContour(self, image):
        contours = ()
        kernel = np.ones((5, 5), np.uint8)
        binary_img = cv2.dilate(image.copy(), kernel, iterations=self.getControlVal('iterations'))
        kernel = np.ones((2, 2), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        dst = binary_img.copy()

        try:
            edged = cv2.Canny(dst, 35, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dst = cv2.dilate(edged.copy(), kernel)  # 扩张

            contours_dilate, hierarchy = cv2.findContours(
                dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            fill_contour_img = np.zeros_like(image)
            cv2.drawContours(fill_contour_img, contours_dilate, -1, (255, 0, 255), -1)  # 轮廓填充

            for contour in contours_dilate:
                dst = cv2.drawContours(dst, contour, -1, (0, 1, 1), 1)
                dst = cv2.fillPoly(dst, [contour], (255, 255, 255))

            contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        except Exception:
            pass
        return contours

    def findStrip(self, contours):
        strip_list = [() for i in range(Strip_DataLength)]
        point_list = [() for i in range(Point_DataLength)]
        if len(contours) != 0:
            energy_strip = self.filterArea(contours, keys=['area_lower_value_energyStrip',
                                                           'area_upper_value_energyStrip'])
            circle_strip = self.filterArea(contours, keys=['area_lower_value_circleStrip',
                                                           'area_upper_value_circleStrip'])
            # assumptionPoint = calMomentCenterPoint(strip_list[Energy_Strip])
            energy_strip_point = calMomentCenter(energy_strip)
            circle_center_point = calMomentCenter(circle_strip)
            strip_list[Energy_Strip], strip_list[Circle_Strip] = energy_strip, circle_strip
            point_list[Energy_Strip_Point], point_list[Circle_Center_Point] = energy_strip_point, circle_center_point
        return strip_list, point_list

    def findTarget(self):
        """
        用以在目标丢失时寻找目标
        """
        pass

    def detectTarget(self, stripList):
        targetRect = []
        if stripList:
            stripList = min(stripList, key=cv2.contourArea)
            targetRect = cv2.minAreaRect(stripList)

        return targetRect

    # def detectTarget(self, strip_list, image):
    #     energy_target_center = ()
    #     energy_target_rect = []
    #     if len(strip_list) != 0:
    #         energy_target, energy_target_rect = self.detectEnergyTarget(strip_list, image.copy())
    #         energy_target_center = calMomentCenter(energy_target)

    # return energy_target_center, energy_target_rect

    def detectEnergyTarget(self, energy_strip_list, image):
        target_list = []
        target_rect = []
        if len(energy_strip_list) != 0:

            fill_contour_img = np.zeros_like(image)
            dilate_contour_img = np.zeros_like(image)
            cv2.drawContours(fill_contour_img, energy_strip_list, -1, (255, 0, 255), -1)  # 轮廓填充
            cv2.drawContours(dilate_contour_img, energy_strip_list, -1, (255, 0, 255),
                             self.getControlVal("Dilate_value"))  # 轮廓
            print(self.getControlVal("Dilate_value"))
            # debug_show_window("fillContourIMG", fillContourIMG)
            # 两张图像相减，获得要击打目标的图像
            target_img = cv2.subtract(fill_contour_img, dilate_contour_img)
            target_img = target_img.astype(np.uint8)
            debug_show_window("EnergyLightStrip", target_img)
            # 调节寻找装甲板大小

            contours_dilate, hierarchy = cv2.findContours(target_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            target_list = self.filterArea(contours_dilate,
                                          keys=['area_lower_value_energyTar', 'area_upper_value_energyTar'])
            if len(target_list) != 0:
                c = min(target_list, key=cv2.contourArea)
                target_rect = cv2.minAreaRect(c)
                # box = cv2.boxPoints(targetRect)  # 转换为long类型
                # box = np.int0(box)
        return target_list, target_rect

    def filterArea(self, contours, keys: list):
        matched_contours_list = []  # 储存所需轮廓列表
        if len(contours) > 0:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                try:
                    if self.getControlVal(keys[0]) < area < self.getControlVal(keys[1]):
                        matched_contours_list.append(contours[i])
                except Exception as e:
                    print(e)
        return matched_contours_list

    def calRotatingCircleCenter(self, centerPoint1, centerPoint2):
        rotating_circle_center = ()
        origin_radian = 0
        if len(centerPoint1) != 0 and len(centerPoint2) != 0:
            origin_radian = calLineRadians(centerPoint1, centerPoint2)  # 计算直线与坐标轴夹角大小（弧度制）
            # print("p1", centerPoint1, "p2", centerPoint2, "degree:", math.degrees(originRadian))
            dis = calPointDistance(centerPoint1, centerPoint2)  # 计算两点距离
            dis *= 4.2
            if dis != 0:
                x = int(centerPoint1[0] + dis * math.cos(origin_radian))
                y = int(centerPoint1[1] + dis * math.sin(origin_radian))
                rotating_circle_center = calAvgPoint((x, y))  # 用以稳定圆心
        return rotating_circle_center, origin_radian

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
        data_list = []
        box = []
        if len(rect) != 0 and rect[0] != ():
            if cvtRectToDataList is True:
                data = [None for i in range(ROI_DATA_LENGTH)]

                data[ROI_RECT] = rect
                data[ROI_BOX] = np.int0(cv2.boxPoints(data[ROI_RECT]))
                data[PNP_INFO] = self.pnp_info(data[ROI_BOX])

                if data[PNP_INFO] is not None:
                    data_list.append(data)
            if cvtRectToBoxPoints is True:
                box = cv2.boxPoints(rect)  # 转换为long类型
                box = np.int0(box)
        if cvtRectToDataList is True:
            return data_list
        elif cvtRectToBoxPoints is True:
            return box
        else:
            return []

    def updateProcess(self, frame, rect=None, points=None):
        img = frame
        if rect is not None:
            if len(rect) != 0:
                Point = rect[0]
                if Point != ():
                    img = self.drawRoi(img.copy(), rect, point=Point)
        if points is not None:
            if len(points) != 0:
                for p in points:
                    if p is not ():
                        img = self.drawRoi(img.copy(), rect=None, point=p)

        debug_show_window("origin img", img)
        return img

    def drawRoi(self, image, rect=None, point=None, color=(255, 0, 255), thickness=2):

        if rect is not None:
            # boxPoints = self.convertDataFormat(rect, cvtRectToDataList=False, cvtRectToBoxPoints=True)
            cv2.drawContours(image, [rect], -1, color, 2)

        if point is not None:
            pointArray = np.array(point, dtype='int16')
            cv2.circle(image, tuple(pointArray), 6, color, -1)

        return image
