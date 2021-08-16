'''
Author: Ligcox
Date: 2021-05-14 21:02:34
LastEditors: Ligcox
LastEditTime: 2021-08-16 15:40:41
Description: Logger.As long as the main use of AOP ideas, recording the various key information in the program.
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import time
import os
import sys
from config.connConfig import *

TEMPORARY_DICT = {
    "yaw": "T0",
    "pitch": "T1",
    "distance": "T2",
    "Kalman_yaw": "T3",
    "Kalman_pitch": "T4",
    "Kalman_distance": "T5",
    "FPS": "T6"
}

log_data = {
    "MainFPS": [0, time.time()],
    "RealsenseFPS": [0, time.time()],
    "mode": [0, time.time()],
    "yaw_angle": [0, time.time()],
    "pitch_angle": [0, time.time()],
    "barrel": [0, time.time()],
    "pathway": [0, time.time()],
    "bcp_frame": [0, time.time()]
}

class BCPloger():
    '''
    BCP协议日志记录器
    生成log文件，每行包含数据项编号、时间、内容
    可使用BCPViewer查看log文件
    '''

    def __init__(self, filename="log/log.log"):
        self.log = open(os.path.join(os.path.split(
            os.path.realpath(__file__))[0], filename), "w")
        self.idx_realsense = 0

    def INFO_LOG(self, name, info):
        '''
        :breif: 记录临时数据，如gimal偏转角度等
        :param: name: 记录数据id
        :param: info: 需要记录的数据信息
        '''
        self.log.write("{},{},{}\n".format(
            TEMPORARY_DICT[name], time.time(), info))

    def BCP_LOG(self, frame):
        '''
        :breif: 记录符合BCP的数据
        :param: frame: BCP数据帧
        '''
        self.log.write("{},{},{}".format(
            TEMPORARY_DICT[name], time.time(), info))

    def RealsenseFPSLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["RealsenseFPS"] = [1/(time.time()-log_data["RealsenseFPS"][1]), time.time()]
            return res
        return wrapper

    def MainFPSLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["MainFPS"] = [1/(time.time()-log_data["MainFPS"][1]), time.time()]
            return res
        return wrapper

    def GimbalLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["yaw_angle"] = [res[0], time.time()]
            log_data["pitch_angle"] = [res[1], time.time()]
            return res
        return wrapper

    def ModeLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["mode"] = [res, time.time()]
            return res
        return wrapper

    def BarrelLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["barrel"] = [res, time.time()]
            return res
        return wrapper

    def PathwayLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["pathway"] = [res, time.time()]
            return res
        return wrapper

    def AngleLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            log_data["_yaw_angle"] = [res[0], time.time()]
            log_data["_pitch_angle"] = [res[1], time.time()]
            return res
        return wrapper

    def PNPLoger(self, func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            if res is None:
                return None
            log_data["PNP_distance"] = [res[0], time.time()]
            log_data["PNP_yaw_angle"] = [res[1], time.time()]
            log_data["PNP_pitch_angle"] = [res[2], time.time()]
            return res
        return wrapper

    def dispaly_loger(self):
        if True:
            sys.stdout.write("{:*^60}\n{:*^60}\n".format("",""))
            sys.stdout.write("{:*^60}\n".format("BCP Viewer Command Version"))
            sys.stdout.write("{:*^60}\n".format("Copyright (c) 2021 Birdiebot R&D department"))
            sys.stdout.write("{:*^60}\n{:*^60}\n\n".format("",""))
            
            sys.stdout.write("{}\n".format("-"*60))
            sys.stdout.write("||{:^15}| {:^14}  | {:^20}|| \n".format("param", "val", "refresh time"))
            sys.stdout.write("{}\n".format("-"*60))

            for key in log_data:
                sys.stdout.write("||{:<15}| {:14.5f}  | {:10.10f}|| \n".format(
                    key, log_data[key][0], log_data[key][1]))

            sys.stdout.write("{}".format("-"*60))

            for i in range(len(log_data)+10):
                sys.stdout.write('\x1b[1A')
                sys.stdout.write("\b"*60)
