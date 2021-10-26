'''
Author: Ligcox
Date: 2021-04-19 20:44:55
LastEditors: Ligcox
LastEditTime: 2021-08-20 01:18:25
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
from targetPredictor import ArmourPredictor
from utils import *

from config.devConfig import source, getThreadingSleepTime
from BCPloger import BCPloger
import logging

class Scheduler(object):
    def __init__(self):
        '''
        description:任务调度器基类，所有机器人调度器应该由该类派生 
        param {*}
        return {*}
        '''        
        self.vIn = vInput(source, False)
        self.conn = Connection()
        self.loger = BCPloger()
        self.mode = 2
    def callback(object):
        '''
        description: 回调函数
        param {*}
        return {*}
        '''
        pass

    def run(self):
        '''
        description: 函数主线程入口，该方法应该由字类重写，使核心任务在的run方法中运行
        param {*}
        return {*}
        '''
        return None

    def __enter__(self):
        '''
        description: 上下文管理器入口
        param {*}
        return {*}
        '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(exc_type, exc_value, traceback.tb_frame)
        '''
        description:上下文管理器出口，该函数会将程序退出时发生的错误打印并执行任务清理 
        param {*}
        return {*}
        '''
        call_cnt = 20
        while call_cnt>0:
            print(str(traceback.tb_frame))
            cur = traceback
            if traceback.tb_next is not None:
                traceback = traceback.tb_next
                call_cnt-=1
        self.cleanup()

    def cleanup(self):
        '''
        description: 结束任务时的清理程序
        param {*}
        return {*}
        '''
        cv2.destroyAllWindows()
        self.vIn.stop()
        self.conn.stop()
        sys.exit(0)


class SentrySchedule(Scheduler):
    def __init__(self):
        '''
        description: 哨兵机器人任务调度器，如果哨兵机器人拥有多云台，请从此类进行派生
        param {*}
        return {*}
        '''
        super().__init__()
        self.cFilter = colorFilter(False)
        self.sDetector = simpleDetector(False)
        self.robot = Sentry_down(self.conn)

    def task_auto_aiming(self, isClassifier=False):
        '''
        description:哨兵自瞄任务 
        param {*isClassifier:是否使用分类器对装甲板进行分类}
        return {*}
        '''
        filtered_frame = self.cFilter.process(self.frame, "hsv")
        armour_list, lightstrips_num = self.sDetector.process(filtered_frame)
        if isClassifier:
            armour_list = self.classifier.process(armour_list, )
        decision_info = self.robot_decision.armour_process(armour_list)
        yaw_angle, pitch_angle, isShoot = decision_info

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
        '''
        description:哨兵下云台任务调度器 
        param {*}
        return {*}
        '''        
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
        '''
        description:哨兵上云台任务调度器
        param {*}
        return {*}
        '''
        super().__init__()
        self.robot = Sentry_up(self.conn)
        self.robot_decision = decision.SentryUpDecision(self.robot)
        from classifier import NumClassifier
        self.classifier = NumClassifier()

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
        '''
        description: 地面机器人任务调度器，其他地面机器人应该从该类派生
        param {*}
        return {*}
        '''
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
        '''
        description: 英雄机器人任务调度器
        param {*}
        return {*}
        '''
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
        '''
        description: 步兵机器人任务调度器
        param {*}
        return {*}
        '''
        self.eFilter = EnergyColorFilter(False)
        self.eDetector = EnergyDetector()
        # self.ePredictor = EnergyPredictor()
        self.eTargetPredictor = ArmourPredictor()
        self.robot_decision = decision.InfantryDecision(self.loger)
        self.robot = Infantry(self.conn)

    def run(self):
        while True:
            self.main_task()

    @loger.MainFPSLoger
    def main_task(self):
        # 心跳数据
        self.robot.heartbeat()
        # 获取串口数据和图像
        self.frame = self.vIn.getFrame()
        if self.frame is not None:
            debug_show_window("Input", self.frame)
            if self.conn.status["mode"] == 0:
                # self.task_auto_aiming()
                self.task_auto_Energy(self.conn.status["pitch_angle"])

            if self.conn.status["mode"] == 2:
                self.task_auto_Energy(self.conn.status["pitch_angle"])

            # loger.dispaly_loger()
            cv2.waitKey(1)

    def task_auto_Energy(self, gimbal_pitch):
        '''
        :brief:根据输入图像自动瞄准能量机关
        '''
        try:
            predict_armour_list = []
            filtered_frame, frame = self.eFilter.process(self.frame)
            # armour_list, data_list = self.eDetector.process(filtered_frame, frame)
            # if data_list is not None:
            #     predict_armour_list = self.ePredictor.process(frame, data_list)
            rect_info = self.eDetector.process(filtered_frame, frame)

            predict_armour_list = self.eTargetPredictor.process(frame, rect_info)
            # decision_info = self.robot_decision.armour_process(predict_armour_list, yaw_angle)
            decision_info = self.robot_decision.energy_process(predict_armour_list , gimbal_pitch)
            yaw_angle, pitch_angle, isShoot = decision_info

            if isShoot == 0xFF:
                cnt = counter(False)
                if cnt == 20:
                    self.robot.mode_ctrl(2)
                    counter(True)

            else:
                counter(True)
                self.robot.mode_ctrl(1)
                self.robot.gimbal(yaw_angle, pitch_angle)
        except Exception as e:
            print(e)

class RadarScheduler(object):
    def __init__(self):
        '''
        description: 雷达任务调度器，这个部分仅仅简单实现功能，后续完善
        param {*}
        return {*}
        '''
        self.vIn = RadarVInput(source, False)

    def run(self):
        while True:
            # 获取串口数据和图像
            self.frame = self.vIn.getFrame()
            if self.frame is not None:
                debug_show_window("Input", self.frame)
                cv2.waitKey(1)