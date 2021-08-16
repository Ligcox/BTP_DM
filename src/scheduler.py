'''
Author: Ligcox
Date: 2021-04-19 20:44:55
LastEditors: Ligcox
LastEditTime: 2021-08-16 14:40:28
Description: Framework core control. Perform the required tasks by BCP according to the different environments on RoboMaster field.
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import decision
from connection import *
from utils import *
from vInput import vInput, RadarVInput
from imageFilter import colorFilter, EnergyColorFilter
from targetDetector import simpleDetector, EnergyDetector
from utils import *

from config.devConfig import source, getThreadingSleepTime
from BCPloger import BCPloger
import logging

class Scheduler(object):
    def __init__(self):
        self.vIn = vInput(source, False)
        self.conn = Connection()

    def callback(object):
        pass

    def run(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(exc_type, exc_value, traceback.tb_frame)
        call_cnt = 20
        while call_cnt>0:
            print(str(traceback.tb_frame))
            cur = traceback
            if traceback.tb_next is not None:
                traceback = traceback.tb_next
                call_cnt-=1
        self.cleanup()

    # 结束任务时的清理程序
    def cleanup(self):
        cv2.destroyAllWindows()
        self.vIn.stop()
        self.conn.stop()
        sys.exit(0)


class SentrySchedule(Scheduler):
    def __init__(self):
        super().__init__()
        self.cFilter = colorFilter(False)
        self.sDetector = simpleDetector(False)
        self.robot = Sentry_down(self.conn)

    def task_auto_aiming(self, isClassifier=False):
        '''
        :breif: 哨兵自瞄任务，在发现目标时对目标进行击打
        '''
        filtered_frame = self.cFilter.process(self.frame, "hsv")
        armour_list, lightstrips_num = self.sDetector.process(filtered_frame)
        if isClassifier:
            armour_list = self.classifier.process(armour_list)
        decision_info = self.robot_decision.armour_process(armour_list)
        yaw_angle, pitch_angle, isShoot = decision_info
        # lightstrips_num, yaw_angle, pitch_angle, isShoot = 2,0,0,1

        if lightstrips_num != 0 and isShoot == 0xFF:
            self.robot.mode_ctrl(2)
            self.robot.barrel(30, 0)
            return
        if lightstrips_num == 0:
            self.robot.barrel(30, 0)
            self.robot.mode_ctrl(0)
            return


class SentryDownScheduler(SentrySchedule):
    def __init__(self):
        super().__init__()
        self.robot = Sentry_down(self.conn)
        self.robot_decision = decision.SentryDownDecision(self.robot)

    def run(self):
        os.system('cls' if os.name == 'nt' else "printf '\033c'")
        while True:
            self.main_task()
            cv2.waitKey(1)
    
    @loger.MainFPSLoger
    def main_task(self):
        # 心跳数据
        self.robot.heartbeat()
        # 获取串口数据和图像
        self.frame = self.vIn.getFrame()
        if self.frame is not None:
            debug_show_window("Input", self.frame)
            # 自瞄模式
            if self.conn.status["mode"] == 0:
                self.task_auto_aiming(isClassifier=False)
        loger.dispaly_loger()

class SentryUpScheduler(SentrySchedule):
    def __init__(self):
        super().__init__()
        self.robot = Sentry_up(self.conn)
        self.robot_decision = decision.SentryUpDecision(self.robot)
        # from classifier import NumClassifier
        # self.classifier = NumClassifier()
        # from autoDodge import AutoDodge
        # self.realsense = AutoDodge(self.robot)

def run(self):
    os.system('cls' if os.name == 'nt' else "printf '\033c'")
    while True:
        self.main_task()
        cv2.waitKey(1)

@loger.MainFPSLoger
def main_task(self):
    # 心跳数据
    self.robot.heartbeat()
    # 获取串口数据和图像
    self.frame = self.vIn.getFrame()
    if self.frame is not None:
        debug_show_window("Input", self.frame)
        # 自瞄模式
        if self.conn.status["mode"] == 0:
            self.task_auto_aiming(isClassifier=False)
    # self.realsense_dispaly_task()
    loger.dispaly_loger()


    def realsense_dispaly_task(self):
        if DEBUG_MODEL:
            try:
                debug_show_window("Realsense", self.realsense.final)
            except Exception as e:
                print(e.args)
                # pass

    # 结束任务时的清理程序
    def cleanup(self):
        cv2.destroyAllWindows()
        self.vIn.stop()
        self.conn.stop()
        self.realsense.stop()
        sys.exit(0)

class GroundSchedule(Scheduler):
    def __init__(self):
        super().__init__()
        self.cFilter = colorFilter(False)
        self.sDetector = simpleDetector(False)

    def task_auto_aiming(self):
        '''
        :breif: 地面机器人自瞄任务，在发现目标时对目标进行击打
        '''
        filtered_frame = self.cFilter.process(self.frame, "hsv")
        armour_list, lightstrips_num = self.sDetector.process(filtered_frame)
        decision_info = self.robot_decision.armour_process(armour_list)
        yaw_angle, pitch_angle, isShoot = decision_info

        if isShoot == 0xFF:
            self.robot.gimbal(0, 0)
            self.robot.mode_ctrl(0)
        else:
            self.robot.mode_ctrl(1)
            self.robot.gimbal(yaw_angle, pitch_angle)

class HeroScheduler(GroundSchedule):
    def __init__(self):
        super().__init__()
        self.robot = Hero(self.conn)
        self.robot_decision = decision.HeroDecision(self.robot)

    def run(self):
        os.system('cls' if os.name == 'nt' else "printf '\033c'")
        while True:
            self.main_task()
            cv2.waitKey(1)
    
    @loger.MainFPSLoger
    def main_task(self):
        # 心跳数据
        self.robot.heartbeat()
        # 获取串口数据和图像
        self.frame = self.vIn.getFrame()
        if self.frame is not None:
            debug_show_window("Input", self.frame)
            # 自瞄模式
            if self.conn.status["mode"] == 0:
                self.task_auto_aiming()

            loger.dispaly_loger()

class InfantryScheduler(GroundSchedule):
    def __init__(self):
        super().__init__()
        self.eFilter = EnergyColorFilter(False)
        self.eDetector = EnergyDetector()

        self.robot_decision = decision.InfantryDecision(self.loger)
        self.robot = Infantry(self.conn)

    def run(self):
        while True:
            # 心跳数据
            self.robot.heartbeat()
            # 获取串口数据和图像
            self.frame = self.vIn.getFrame()
            if self.frame is not None:
                debug_show_window("Input", self.frame)
                # mode_ctrl = self.mode_process(self.conn.rx_info)
                # 自瞄模式
                # print("mode", self.conn.status["mode"])
                if self.conn.status["mode"] == 0:
                    self.task_auto_aiming()
                    # self.task_auto_Energy()
                if self.conn.status["mode"] == 2:
                    self.task_auto_Energy()
                cv2.waitKey(1)

    def task_auto_Energy(self):
        '''
        :breif:根据输入图像自动瞄准能量机关
        '''
        filtered_frame, frame = self.eFilter.process(self.frame)
        armour_list = self.eDetector.process(filtered_frame, frame)
        decision_info = self.robot_decision.armour_process(armour_list)
        yaw_angle, pitch_angle, isShoot = decision_info

        if isShoot == 0xFF:
            self.robot.gimbal(0, 0)
            self.robot.mode_ctrl(0)
        else:
            self.robot.mode_ctrl(1)
            self.robot.gimbal(yaw_angle, pitch_angle)

class RadarScheduler(object):
    def __init__(self):
        self.vIn = RadarVInput(source, False)

    def run(self):
        while True:
            # 获取串口数据和图像
            self.frame = self.vIn.getFrame()
            if self.frame is not None:
                debug_show_window("Input", self.frame)
                cv2.waitKey(1)