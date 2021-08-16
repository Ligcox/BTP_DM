'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-10 15:44:40
Description: Target detector. Instantiate different classes to detect different targets in different scenarios.
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
from module import *
from utils import *
from config.config import *

# 相机标定矩阵
Camera_intrinsic = {
    "mtx": np.array([[1.32280384e+03, 0.00000000e+00, 3.09015137e+02],
                     [0.00000000e+00, 1.32691122e+03, 2.04586526e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                    dtype=np.double),
    "dist": np.array([2.50140551e-03, -3.05940539e+00, -9.93333169e-03,
                      -3.64458084e-04, 2.33302573e+01], dtype=np.double),
}


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
                strip_data = [None for i in range(ROI_DATALENTH)]
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
                        armor_data = [None for i in range(ROI_DATALENTH)]
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

    @loger.PNPLoger
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

    def updateProcess(self, frame, lightstrips=[], armors=[]):
        if not (self.getControlVal('silent') or (frame is None)):
            img = frame.copy()
            cv2.putText(img, 'ROI', (0, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            img0 = self.drawRoi(img, lightstrips, self.getControlVal(
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

        super().__init__(hide_controls)

    def process(self, image, frame):
        """
        :param image:
        :param frame:
        :return:
        energyStrip：锤子形轮廓
        assumptionPoint：假设点为锤子形轮廓的几何矩
        energyTarget：击打目标
        tarCenterPoint：击打目标中心点
        circleCenterPoint：旋转圆心点
        originRadian：assumptionPoint与tarCenterPoint两点关于x轴的夹角
        predictPoint：预测点
        predictRect：预测矩形（用以pnp解算）
        """
        energyTargetList = []
        energyStrip, assumptionPoint = self.detectEnergyLightStrip(image.copy())

        energyTarget, tarCenterPoint = self.detectEnergyTarget(energyStrip, image.copy())

        circleCenterPoint, originRadian = self.calRotatingCircleCenter(assumptionPoint, tarCenterPoint)

        predictPoint = self.predictTarPoint(0.4, circleCenterPoint, tarCenterPoint, originRadian)  # 预测击打点和能量机关圆心

        predictRect, energyTargetList = self.predictTarRect(predictPoint, energyTarget)  # 预测击打点的矩形框

        # self.updateProcess(frame, (energyTarget[0][ROI_BOX],predictRect), (predictPoint, circleCenterPoint, centerPoint1))
        self.updateProcess(frame, predictRect, (predictPoint, circleCenterPoint, tarCenterPoint))

        # self.updateProcess(frame, energyTarget, circleCenterPoint)

        return energyTargetList

    def detectEnergyLightStrip(self, image):
        centerPoint1 = ()
        debug_show_window("EnergyLightStrip", image)

        # image=cv2.GaussianBlur(image,(3,3),0)
        kernel = np.ones((5, 5), np.uint8)
        BinaryImg = cv2.dilate(image.copy(), kernel, iterations=self.getControlVal('iterations'))
        kernel = np.ones((2, 2), np.uint8)
        # BinaryImg = cv2.morphologyEx(BinaryImg, cv2.MORPH_OPEN, kernel, iterations=5)
        BinaryImg = cv2.morphologyEx(BinaryImg, cv2.MORPH_CLOSE, kernel, iterations=1)
        dst = BinaryImg.copy()

        EnergyStripList = []  # 储存所需轮廓列表
        try:
            edged = cv2.Canny(dst, 35, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dst = cv2.dilate(edged.copy(), kernel)  # 扩张

            contours_dilate, hierarchy = cv2.findContours(
                dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            fillContourIMG = np.zeros_like(image)
            cv2.drawContours(fillContourIMG, contours_dilate, -1, (255, 0, 255), -1)  # 轮廓填充

            for contour in contours_dilate:
                dst = cv2.drawContours(dst, contour, -1, (0, 1, 1), 1)
                dst = cv2.fillPoly(dst, [contour], (255, 255, 255))

            contours, hierarchy = cv2.findContours(
                dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if self.getControlVal('area_lower_value') < area < self.getControlVal('area_upper_value'):
                    EnergyStripList.append(contours[i])

                    # cv2.circle(image, (np.int(cx), np.int(cy)), 3, (0, 0, 255), -1)

            if len(EnergyStripList) != 0:
                c = min(EnergyStripList, key=cv2.contourArea)

                mm = cv2.moments(c)  # 求轮廓的几何矩
                cx = mm['m10'] / mm['m00']  # 原点的零阶矩
                cy = mm['m01'] / mm['m00']
                # print("cx",cx)
                # marker = cv2.minAreaRect(c)
                # box = np.int0(cv2.boxPoints(marker))

                centerPoint1 = (int(cx), int(cy))

        except Exception:
            pass
        return EnergyStripList, centerPoint1

    def detectEnergyTarget(self, EnergyStripList, image):
        Energy_target_data = [None for i in range(ROI_DATALENTH)]
        Energy_target_list = []
        targetList = []
        centerPoint2 = ()
        fillContourIMG = np.zeros_like(image)
        dilateContourIMG = np.zeros_like(image)
        cv2.drawContours(fillContourIMG, EnergyStripList, -1, (255, 0, 255), -1)  # 轮廓填充
        cv2.drawContours(dilateContourIMG, EnergyStripList, -1, (255, 0, 255),
                         self.getControlVal("Dilate_value"))  # 轮廓
        debug_show_window("fillContourIMG", fillContourIMG)
        # 两张图像相减，获得要击打目标的图像
        targetIMG = cv2.subtract(fillContourIMG, dilateContourIMG)
        targetIMG = targetIMG.astype(np.uint8)
        debug_show_window("EnergyLightStrip", targetIMG)
        # 调节寻找装甲板大小
        contours_dilate, hierarchy = cv2.findContours(
            targetIMG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours_dilate):
            area = cv2.contourArea(contour)
            # print(area)
            if self.getControlVal('area_lower_value1') < area < self.getControlVal('area_upper_value1'):
                targetList.append(contour)

        if len(targetList) != 0:
            c = min(targetList, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)  # 转换为long类型
            box = np.int0(box)

            # height = abs(box[0][1] - box[2][1])  # 轮廓长度
            # weight = abs(box[0][0] - box[2][0])  # 轮廓宽度
            height = calPointDistance(box[0], box[2])  # 轮廓长度
            weight = calPointDistance(box[0], box[1])  # 轮廓宽度
            # print("height:", height, "weight:", weight, "b0", box[0], "b1", box[1], "b2", box[2])
            if height > weight:
                height, weight = weight, height
            if height != 0:
                ratio = float(weight) / float(height)  # 长宽比

                if ratio < 3.5:
                    Energy_target_data[ROI_RECT] = rect
                    Energy_target_data[ROI_BOX] = np.int0(cv2.boxPoints(Energy_target_data[ROI_RECT]))
                    Energy_target_data[PNP_INFO] = self.pnp_info(Energy_target_data[ROI_BOX])
                    if Energy_target_data[PNP_INFO] is not None:
                        Energy_target_list.append(Energy_target_data)
                        box = Energy_target_list[0][ROI_BOX]
                        centerPoint2 = calCenterCoordinate(box)
        return Energy_target_list, centerPoint2

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

    def calRotatingCircleCenter(self, centerPoint1, centerPoint2):
        RotatingCircleCenter = ()
        originRadian = 0
        if len(centerPoint1) != 0 and len(centerPoint2) != 0:
            originRadian = calLineRadians(centerPoint1, centerPoint2)  # 计算直线与坐标轴夹角大小（弧度制）
            # print("p1", centerPoint1, "p2", centerPoint2, "degree:", math.degrees(originRadian))
            dis = calPointDistance(centerPoint1, centerPoint2)  # 计算两点距离
            dis *= 3.4
            if dis != 0:
                x = int(centerPoint1[0] + dis * math.cos(originRadian))
                y = int(centerPoint1[1] + dis * math.sin(originRadian))
                RotatingCircleCenter = calAvgPoint((x, y))  # 用以稳定圆心
        return RotatingCircleCenter, originRadian

    def predictTarPoint(self, time, RotatingCircleCenter, tarCenterPoint, originRadian):
        predictPoint = ()

        if len(RotatingCircleCenter) != 0 and len(tarCenterPoint) != 0:
            x, y = RotatingCircleCenter
            displacement = calDisplacement(tarCenterPoint)  # 计算上一帧与这一帧目标中心点位移

            if displacement > 0:
                timeDiffer = calTimeDiffer()  # 计算两帧的时间差
                # print(displacement)

                if timeDiffer != 0:

                    Velocity = displacement / timeDiffer  # 计算目标中心点移动速度
                    if displacement > 80:
                        Velocity = calAvgVelocity(Velocity, True)
                    else:
                        Velocity = calAvgVelocity(Velocity)

                    radius = calPointDistance(RotatingCircleCenter, tarCenterPoint)  # 目标中心与旋转中心的距离

                    S = Velocity * time
                    predictAngle = 180 * S / (math.pi * radius)
                    if predictAngle > 60:
                        predictAngle = 60
                    predictRadian = math.radians(predictAngle)

                    finalRadian = originRadian - predictRadian

                    predictX = int(x - radius * math.cos(finalRadian))
                    predictY = int(y - radius * math.sin(finalRadian))

                    predictPoint = (predictX, predictY)
            return predictPoint

    def predictTarRect(self, center, energyTarget):
        Rect = []
        energyTargetData = [None for i in range(ROI_DATALENTH)]
        energyTargetList = []
        print(energyTarget,center)
        if len(energyTarget) != 0 and len(center) != 0:
            print(8)
            rect = energyTarget[ROI_RECT]
            centerX, centerY = center

            weight = rect[ROI_RECT][1][0]
            height = rect[ROI_RECT][1][1]
            angel = rect[ROI_RECT][2]
            angelPi = (angel / 180) * math.pi

            x1 = int(centerX + (weight / 2) * math.cos(angelPi) - (height / 2) * math.sin(angelPi))
            y1 = int(centerY + (weight / 2) * math.sin(angelPi) + (height / 2) * math.cos(angelPi))

            x2 = int(centerX + (weight / 2) * math.cos(angelPi) + (height / 2) * math.sin(angelPi))
            y2 = int(centerY + (weight / 2) * math.sin(angelPi) - (height / 2) * math.cos(angelPi))

            x3 = int(centerX - (weight / 2) * math.cos(angelPi) + (height / 2) * math.sin(angelPi))
            y3 = int(centerY - (weight / 2) * math.sin(angelPi) - (height / 2) * math.cos(angelPi))

            x4 = int(centerX - (weight / 2) * math.cos(angelPi) - (height / 2) * math.sin(angelPi))
            y4 = int(centerY - (weight / 2) * math.sin(angelPi) + (height / 2) * math.cos(angelPi))

            Rect = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            Rect2 = (center, (weight, height), angel)
            energyTargetData[ROI_RECT] = Rect2
            energyTargetData[ROI_BOX] = np.int0(cv2.boxPoints(energyTargetData[ROI_RECT]))
            energyTargetData[PNP_INFO] = self.pnp_info(energyTargetData[ROI_BOX])

            if energyTargetData[PNP_INFO] is not None:
                energyTargetList.append(energyTargetData)

        return Rect, energyTargetList

    def updateProcess(self, frame, box=None, points=None):
        img = frame
        if len(box) != 0:
            Point = calCenterCoordinate(box)
            img = self.drawRoi(img.copy(), box, point=Point)
        if len(points) != 0:
            for p in points:
                if p is not ():
                    img = self.drawRoi(img.copy(), box=None, point=p)

        debug_show_window("targetImg", img)
        return img
        # 卡尔曼滤波

        # self.kalman_func(int(box[0][0] + (box[2][0] - box[0][0]) / 2), int(box[0][1] + (box[2][1] - box[0][1]) / 2))

    def drawRoi(self, image, box: None, point=None, color=(255, 0, 255), thickness=2):

        if box is not None:
            cv2.drawContours(image, [box], -1, color, 2)

        if point is not None:
            cv2.circle(image, point, 3, color, -1)

        return image
