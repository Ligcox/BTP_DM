'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-20 16:15:36
Description: Program decision level, all robot decision information should be processed by this module and then sent.
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import math

from module import *
from utils import *
from config.config import *


class Decision(module):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 决策层类
        param {*}
        return {*}
        '''
        super().__init__(hide_controls)
        self.robot = robot
        self.armour_time_queue = Queue()
        self.last_yaw_angle, self.last_pitch_angle = 0, 0

    def empty_disable_time(self, disableTime=1):
        '''
        :brief: 清除超过disableTime的时间，防止queue无限扩大
        :param: disableTime:清除无效时间的间隔
        '''
        now_time = time.time()
        if not self.armour_time_queue.empty():
            try:
                while now_time - self.armour_time_queue.queue[0] >= disableTime:
                    self.armour_time_queue.get(False)
            except:
                pass
        else:
            self.last_yaw_angle, self.last_pitch_angle = 0, 0
        return None

    def pnp_error_compensation(self, ROI_RECT, distance):
        '''
        description: pnp解算远距离是预测位置偏下额外做补偿
        param {*ROI_RECT: 装甲板RECT信息, distance: 装甲板距离}
        return {*}
        '''
        w, h = 640, 480
        x, y = ROI_RECT[0]
        x, y = x-w/2, y-h/2
        # x = x*distance*0.001
        # print(y, distance)
        if distance > 3000:
            x = 0
            y = distance*self.getControlVal("pitch_pnp_error")/1000
        else:
            x, y = 0, 0
        return x, y

    def differential_filter(self, yaw_speed, pitch_spend):
        '''
        @description: 通过两次目标之间的差分获得此次击打目标的提前量
        @param {*yaw_speed: 当前yaw信息, *pitch_spend: 当前pitch信息}
        @return {*}
        '''
        d_y = yaw_speed - (self.last_yaw_angle)
        d_p = pitch_spend - (self.last_pitch_angle)
        d_y = abs_min_filter(d_y, 0.3)
        d_p = abs_min_filter(d_p, 0.3)
        d_y = abs_max_filter(d_y, 1)
        d_p = abs_max_filter(d_p, 1)
        yaw_speed += d_y
        pitch_spend += d_p
        return yaw_speed, pitch_spend

    def gimbal_send(self, mode, yaw_angle, pitch_angle, isShoot):
        '''
        description: 将yaw_angle, pitch_angle, isShoot三个数据打包直接发送
        param {*}
        return {*}
        '''
        self.robot.mode_ctrl(mode)
        self.robot.gimbal(yaw_angle, pitch_angle)
        self.robot.barrel(30, isShoot)


class SentryDecision(Decision):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 哨兵决策层，若有多个云台应该由该类派生
        param {*}
        return {*}
        '''
        super().__init__(robot, hide_controls=hide_controls)

    def armour_process(self, armour_list):
        '''
        @description: 装甲板识别任务，取出最近的装甲板作为击打的对象
        @param {*}
        @return {*}yaw、pitch偏转角度，枪口是否发射
        '''
        yaw_angle, pitch_angle, isShoot = 0, 0, 0
        # 先清除失效时间
        self.empty_disable_time()

        if len(armour_list) != 0:
            # 寻找装甲板列表中最近的装甲板
            def f(x): return x[-1][0]
            armour_list.sort(key=f)
            ROI_RECT, ROI_BOX, PNP_LIST = armour_list[0]
            distance, yaw_angle, pitch_angle = PNP_LIST

            # 将最后发现装甲板的时间点存入时间序列、
            self.armour_time_queue.put(time.time())

            # 弹道补偿
            # 远距离给予额外的控制量
            yaw_angle_error, pitch_angle_error = self.pnp_error_compensation(
                ROI_RECT, distance)
            yaw_angle += yaw_angle_error
            pitch_angle += pitch_angle_error

            yaw_angle += self.getControlVal("yaw_angle_offset")
            pitch_angle += self.getControlVal("pitch_angle_offset")

            yaw_angle = abs_max_filter(yaw_angle, 3)
            pitch_angle = abs_max_filter(pitch_angle, 3)

            # 一秒内发现五帧目标
            if self.armour_time_queue.qsize() >= 5:
                yaw_angle, pitch_angle = self.differential_filter(
                    yaw_angle, pitch_angle)
                self.gimbal_send(1, yaw_angle, pitch_angle, 1)

                self.last_yaw_angle = yaw_angle
                self.last_pitch_angle = pitch_angle
            else:
                self.gimbal_send(1, yaw_angle, pitch_angle, 0)

        # 未发现装甲板
        else:
            # 由于击打装甲板闪烁无法找到装甲板
            if self.armour_time_queue.qsize() >= 30:
                self.gimbal_send(1, self.last_yaw_angle,
                                 self.last_pitch_angle, 1)
            # 未发现目标，由下位机接管或进入微调模式
            else:
                isShoot = 0xFF

        return yaw_angle, pitch_angle, isShoot


class SentryDownDecision(SentryDecision):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 哨兵下云台决策层
        param {*}
        return {*}
        '''
        self.controls = sentryDown_decision_controls
        self.name = "sentryDown_decision"
        super().__init__(robot, hide_controls)


class SentryUpDecision(SentryDecision):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 哨兵上云台决策层
        param {*}
        return {*}
        '''
        self.controls = sentryUp_decision_controls
        self.name = "sentryUp_decision"
        super().__init__(robot, hide_controls)


class GroundDecison(Decision):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 地面机器人决策层，其他地面机器人应该由该类派生
        param {*}
        return {*}
        '''
        super().__init__(robot, hide_controls=hide_controls)

    def armour_process(self, armour_list):
        '''
        :breif: 装甲板识别任务，取出最近的装甲板作为击打的对象
        :return: yaw、pitch偏转角度，枪口是否发射
        '''
        yaw_angle, pitch_angle, isShoot = 0, 0, 0

        yaw_angle_offset = self.getControlVal("yaw_angle_offset")
        pitch_angle_offset = self.getControlVal("pitch_angle_offset")

        if len(armour_list) != 0:
            # 寻找装甲板列表中最靠近中心的装甲板
            def f(x):
                return (x[-1][1]-yaw_angle_offset)**2 + (x[-1][2]-pitch_angle_offset)**2
            armour_list.sort(key=f)
            ROI_RECT, ROI_BOX, PNP_LIST = armour_list[0]
            distance, yaw_angle, pitch_angle = PNP_LIST

            yaw_angle += yaw_angle_offset
            pitch_angle += pitch_angle_offset

            yaw_angle = abs_max_filter(yaw_angle, 3)
            pitch_angle = abs_max_filter(pitch_angle, 3)
            isShoot = 1

            self.last_yaw_angle = yaw_angle
            self.last_pitch_angle = pitch_angle

        # 未发现目标，由下位机接管
        else:
            isShoot = 0xFF
            self.last_yaw_angle = 0xFFFF
            self.last_pitch_angle = 0xFFFF

        return yaw_angle, pitch_angle, isShoot


class HeroDecision(GroundDecison):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 英雄机器人决策层
        param {*}
        return {*}
        '''
        self.controls = hero_decision_controls
        self.name = "hero_decision"
        self.armour_time_queue = Queue()
        super().__init__(robot, hide_controls)


class InfantryDecision(GroundDecison):
    def __init__(self, robot, hide_controls=False):
        '''
        description: 步兵机器人决策层
        param {*}
        return {*}
        '''
        self.controls = decision_controls
        self.name = "infantry_decision"
        self.armour_time_queue = Queue()
        super().__init__(robot, hide_controls)

    def energy_process(self, pnp_list, gimbal_pitch):
        '''
        :brief: 能量机关识别任务，识别点亮的能量机关
        :return: yaw、pitch偏转角度
        '''
        yaw_angle, pitch_angle, isShoot = 0, 0, 0
        # 先清除失效时间
        self.empty_disable_time()

        if len(pnp_list) != 0:
            distance, yaw_angle, pitch_angle = pnp_list
            adjust_angle = self.adjustBallistics(distance,gimbal_pitch=45,ball_speed=15)
            yaw_angle += self.getControlVal("yaw_angle_offset")
            pitch_angle = self.getControlVal("pitch_angle_offset") + yaw_angle

            isShoot = 1

        # 未发现目标，由下位机接管
        else:
            isShoot = 0
        return yaw_angle, pitch_angle, isShoot

    def adjustBallistics(self, distance, gimbal_pitch,ball_speed):
        g = 9.778
        ration = math.radians(gimbal_pitch)
        Z = distance * math.cos(ration)
        Y = distance * math.sin(ration)
        for i in range(20):
            down_time = distance / 800.0 / ball_speed  # 下坠时间
            offset_gravity = 0.5 * g * down_time ** 2 * 1000  # 下落距离
            new_angle = math.atan((Y+offset_gravity)/Z)
            distance = abs(Z/math.cos(new_angle))
        adjust_angle = math.degrees(new_angle)-gimbal_pitch
        return adjust_angle
