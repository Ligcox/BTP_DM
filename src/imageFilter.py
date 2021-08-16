'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-10 15:18:36
Description: Color filter, through the inheritance of Class 'imageFilter', realize the color filtering in different scenarios
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
from module import *
from utils import *
from config.config import *


class imageFilter(module):
    name = "Empty Filter Control"

    def process(self, frame):
        return frame


class colorFilter(imageFilter):
    def __init__(self, hide_controls=True):
        self.controls = channel_filter_controls
        self.name = "ColorFilter"
        super().__init__(hide_controls)

    def process(self, frame, colorSpace):
        # Hard code channel number
        NUM_CHANNEL = 3
        if not frame.shape[2] == NUM_CHANNEL:
            # deal with BGR color frames only
            return frame

        # 判断使用hsv或rgb颜色空间
        if colorSpace == "rgb":
            target_color = [self.getControlVal('blueTarget'), self.getControlVal(
                'greenTarget'), self.getControlVal('redTarget')]
            tolerance = [self.getControlVal('blueTolerance'), self.getControlVal(
                'greenTolerance'), self.getControlVal('redTolerance')]
        elif colorSpace == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            target_color = [self.getControlVal('hueTarget'), self.getControlVal(
                'saturationTarget'), self.getControlVal('valueTarget')]
            tolerance = [self.getControlVal('hueTolerance'), self.getControlVal(
                'saturationTolerance'), self.getControlVal('valueTolerance')]
        else:
            raise AttributeError("Only use RGB or HSV color space.")

        channel_target = [
            [
                max(0, target_color[i]-tolerance[i]),
                min(255, target_color[i]+tolerance[i])
            ]
            for i in range(NUM_CHANNEL)]
        channel_masks = []
        final_mask = np.ones(frame.shape[:2], np.uint8)*255
        methods = self.getControlVal('maskMethod')
        for channel in range(NUM_CHANNEL):
            _, low_mask = cv2.threshold(
                frame[:, :, channel], channel_target[channel][0], 255, methods[0])
            _, high_mask = cv2.threshold(
                frame[:, :, channel], channel_target[channel][1], 255, methods[1])
            channel_masks.append(cv2.bitwise_and(low_mask, high_mask))
            if channel == 10:
                continue
            final_mask = cv2.bitwise_and(final_mask, channel_masks[channel])
        output = cv2.bitwise_and(frame, frame, mask=final_mask)
        self.updateProcess(final_mask, channel_masks, colorSpace)
        return output

    def updateProcess(self, final_mask, channel_masks, colorSpace):
        if not (self.getControlVal('silent') or (final_mask is None)):
            if colorSpace == "rgb":
                blueMask = channel_masks[0].copy()
                cv2.putText(blueMask, 'BlueChannelMask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                greenMask = channel_masks[1].copy()
                cv2.putText(greenMask, 'GreenChannelMask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                redMask = channel_masks[2].copy()
                cv2.putText(redMask, 'RedChannelMask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                finalMask = final_mask.copy()
                cv2.putText(finalMask, 'FinalMask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                row1 = np.concatenate((finalMask, blueMask), axis=1)
                row2 = np.concatenate((greenMask, redMask), axis=1)
            elif colorSpace == "hsv":
                HueMask = channel_masks[0].copy()
                cv2.putText(HueMask, 'Hue Channel Mask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                SaturationMask = channel_masks[1].copy()
                cv2.putText(SaturationMask, 'Saturation Channel Mask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                ValueMask = channel_masks[2].copy()
                cv2.putText(ValueMask, 'Value Channel Mask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                finalMask = final_mask.copy()
                cv2.putText(finalMask, 'FinalMask', (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

                row1 = np.concatenate((finalMask, HueMask), axis=1)
                row2 = np.concatenate((SaturationMask, ValueMask), axis=1)
            img = np.concatenate((row1, row2), axis=0)
        else:
            img = BLANK_IMG
        debug_show_window(self.control_window_name + "img", img)


class EnergyColorFilter(imageFilter):
    def __init__(self, hide_controls=True):
        self.controls = energy_filter_controls
        self.name = "EnergyFilter"
        super().__init__(hide_controls)

    def process(self, frame):
        # frame = aug(frame, self.getControlVal("aug"))
        grayImage = self.splitChannel(frame)
        binaryImgByRGB = self.RGBFilter(grayImage)
        binaryImgByHSV = self.HSVFilter(frame)

        # kernel = np.ones((1, 1), np.uint8)
        # binaryImgByHSV = cv2.erode(binaryImgByHSV, kernel, iterations=self.getControlVal('iterations'))
        finalMask = cv2.bitwise_and(binaryImgByRGB, binaryImgByHSV)
        # finalMask=cv2.GaussianBlur(finalMask, (3, 3), 0)
        self.updateProcess("finalMask", finalMask)
        return finalMask, frame




    def splitChannel(self, frame):
        # 分离通道
        bChannel, gChannel, rChannel = cv2.split(frame)
        return gChannel

    def RGBFilter(self, grayImage):
        """
        RGB空间滤波器
        :param grayImage: 灰度图
        :return: 二值图
        """
        ret, binaryImgByRGB = cv2.threshold(grayImage, self.getControlVal('threshold'), 255, cv2.THRESH_BINARY)  # 二值化图像
        self.updateProcess("BinaryImgByRGB", binaryImgByRGB, True)
        return binaryImgByRGB

    def HSVFilter(self, frame):
        """
        HSV空间滤波器
        :param frame: 三通道BGR图
        :return: 二值图
        """
        hsvImg = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)  # 颜色空间转换
        lower = [self.getControlVal("lowHue"), self.getControlVal("lowSat"), self.getControlVal("lowVal")]
        upper = [self.getControlVal("highHue"), self.getControlVal("highSat"), self.getControlVal("highVal")]
        color_dist = {1: {'Lower': np.array(lower), 'Upper': np.array(upper)}}
        binaryImgByHSV = cv2.inRange(hsvImg, color_dist[1]['Lower'], color_dist[1]['Upper'])
        # binaryImgByHSV = cv2.bitwise_not(binaryImgByHSV)
        self.updateProcess("BinaryImgByHSV", binaryImgByHSV, True)
        return binaryImgByHSV

    def updateProcess(self, windowName: str, BinaryImg, hideWindow=False):
        if hideWindow is False:
            debug_show_window(f"{windowName}", BinaryImg)
