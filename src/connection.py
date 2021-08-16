'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-12 01:54:44
Description: The principal implementation of Birdiebot Communication Protocol
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
from config import *
from utils import *
from config.devConfig import *
from config.connConfig import *

class Connection(object):
    def __init__(self, p=PORTX, b=BPS, to=TIMEX):
        self.device = serial.Serial(port=p, baudrate=b, timeout=to)
        self.stop_flag = False

        self.status = STATUS

        self.tx_queue = []
        self.tx_thread = threading.Thread(target=self.tx_function, name="tx_thread")

        self.reset_rx_buffer()
        self.rx_queue = Queue()
        self.rx_thread = threading.Thread(target=self.receive, name="rx_thread")

        self.current_packet = copy.deepcopy(D_INFO)
        self.start()

    def start(self):
        self.stop_flag = False
        self.rx_thread.start()
        self.tx_thread.start()

    def stop(self):
        self.stop_flag = True
        self.rx_thread.join()

    def reset_rx_buffer(self):
        self.current_packet = copy.deepcopy(D_INFO)
        self.rx_status = 0
        self.rx_datalen = 0

    def rx_function(self):
        rx_bytes = self.device.readall()
        # print(len(rx_bytes))
        for rx_byte in rx_bytes:
            if self.rx_status == 0:  # 等待HEAD
                if rx_byte == D_INFO["HEAD"]:
                    self.rx_status = 1
            elif self.rx_status == 1:  # 等待D_ADDR
                if rx_byte ==  D_ADDR["mainfold"]:
                # if rx_byte ==  D_ADDR["infantry"]:
                    self.current_packet["D_ADDR"] = rx_byte
                    self.rx_status = 2
                else:
                    self.reset_rx_buffer()
            elif self.rx_status == 2:  # 等待ID
                self.current_packet["ID"] = rx_byte
                self.rx_status = 3
            elif self.rx_status == 3:  # 等待LEN
                # 貌似python3直接转换byte->int，有问题自己纠正
                self.current_packet["LEN"] = rx_byte
                if rx_byte == 0:
                    self.rx_status = 5
                else:
                    self.rx_status = 4
            elif self.rx_status == 4:  # 等待DATA
                self.current_packet["DATA"].append(rx_byte)
                self.rx_datalen += 1
                if self.rx_datalen >= self.current_packet["LEN"]:
                    self.rx_status = 5
            elif self.rx_status == 5:  # 等待SUM_CHECK
                self.current_packet["SUM_CHECK"], self.current_packet["ADD_CHECK"] = sumcheck_cal(self.current_packet)
                if rx_byte == self.current_packet["SUM_CHECK"]:
                    self.rx_status = 6
                else:  # 校验失败
                    self.reset_rx_buffer()
            elif self.rx_status == 6:  # 等待ADD_CHECK
                if rx_byte == self.current_packet["ADD_CHECK"]:
                    self.rx_queue.put(copy.deepcopy(self.current_packet))
                    self.bcpAnalysis()
                    # self.id2 += 1
                    # print(self.id2, self.current_packet)
                self.reset_rx_buffer()  # 校验失败或者成功后都需要重设

    @loger.AngleLoger
    def gimbalAnalysis(self, bcpframe):
        self.status["yaw_angle"] = struct.pack("h", bcpframe["DATA"][0], bcpframe["DATA"][1]) /1000
        self.status["pitch_angle"] = struct.pack("h", bcpframe["DATA"][2], bcpframe["DATA"][3]) /1000
        return self.status["yaw_angle"], self.status["pitch_angle"]


    def bcpAnalysis(self):
        bcpframe = self.rx_queue.get(False)
        if bcpframe["ID"] == ID["manifold_ctrl"]:
            self.status["mode"] = bcpframe["DATA"][0]
        elif bcpframe["ID"] == ID["barrel"]:
            _val = bcpframe["DATA"][0]
            self.status["isShoot"] = _val
        elif bcpframe["ID"] == ID["chassis"]:
            _val = bcpframe["DATA"][0]
            self.status["pathway_direction"] = _val
        elif bcpframe["ID"] == ID["chassis_speed"]:
            _val = struct.pack("i", bcpframe["DATA"][0], bcpframe["DATA"][1], bcpframe["DATA"][2],bcpframe["DATA"][3])
            self.status["pathway_speed"] = _val
        elif bcpframe["ID"] == ID["gimbal_angle"]:
            self.gimbalAnalysis(bcpframe)

    def tx_function(self):
        while not self.stop_flag:
            while len(self.tx_queue) != 0:
                tx_packet = self.tx_queue.pop()
                # print(tx_packet)
                # if tx_packet[2] == 0x02:
                #     print("gimbal", tx_packet)
                # if tx_packet[2] == 0x06:
                    # print( "mode ctrl",tx_packet[3])
                # if tx_packet[2] == 0xAA:
                #     print( "heartbeat",tx_packet)
                self.device.write(tx_packet)
            time.sleep(getThreadingSleepTime("tx_threading"))

    def send(self, tx_packet):
        self.tx_queue.append(copy.deepcopy(tx_packet))

    def receive(self):
        self.id2 = 0
        while not self.stop_flag:
            self.rx_function()
            time.sleep(getThreadingSleepTime("rx_threading"))

class SerialInfo(object):
    def __init__(self):
        self.INFO = copy.deepcopy(D_INFO)

    def getInfo(self):
        return bytearray([
            self.INFO["HEAD"],
            self.INFO["D_ADDR"],
            self.INFO["ID"],
            self.INFO["LEN"],
            *self.INFO["DATA"],
            self.INFO["SUM_CHECK"],
            self.INFO["ADD_CHECK"]
        ])


class Robot(SerialInfo):
    def __init__(self, conn: Connection, name=None, data=None):
        super().__init__()
        self.name = name
        self.initRobot()
        self.conn = conn
        if data is not None:
            self.setDATA(*data)
        # 心跳帧变量设置
        self.beat_info = 0
        self.last_time = 0
        self.status = conn.status

    def __call__(self, idx, identif, data):
        self.setID(idx)
        self.setDATA(identif, data)
        return self.getInfo()

    def setID(self, idx):
        self.INFO["ID"] = ID[idx]

    def setDATA(self, identif, data):
        # print(data)
        identif = "<" + identif
        if isinstance(data, int):
            self.INFO["DATA"] = struct.pack(identif, data)
            self.INFO["LEN"] = 1
        else:
            self.INFO["DATA"] = struct.pack(identif, *data)
            self.INFO["LEN"] = len(self.INFO["DATA"])
            
        self.INFO["SUM_CHECK"], self.INFO["ADD_CHECK"] = sumcheck_cal(self.INFO)

    def initRobot(self):
        self.INFO["D_ADDR"] = D_ADDR[self.name]

    @loger.ModeLoger
    def mode_ctrl(self, stu):
        '''
        @brief: 模式控制
        发送任意值，使得哨兵模式更改
        # TODO
        '''
        self.setID("mode")
        self.setDATA("B", stu)
        self.conn.send(self.getInfo())
        return stu

    def launch(self):
        self.setDATA("", self.INFO)

    @loger.GimbalLoger
    def gimbal(self, yaw_angle, pitch_angle):
        '''
        @brief: 控制云台偏转
        '''
        self.setID("gimbal")
        self.setDATA("hh", (int(yaw_angle*1000), int(pitch_angle*1000)))
        self.conn.send(self.getInfo())
        return yaw_angle, pitch_angle
    
    @loger.BarrelLoger
    def barrel(self, speed, stu):
        '''
        @brief: 枪管发射
        stu: 0为发射，1为不发射
        '''
        self.setID("barrel")
        self.setDATA("BB", (speed, stu))
        self.conn.send(self.getInfo())
        return stu

    def heartbeat(self):
        '''
        :brief: 心跳数据,每隔50ms
        '''
        if time.time() - self.last_time >= 0.05:
            self.setID("heartbeat")
            self.beat_info = 0 if self.beat_info == 1 else 1
            self.setDATA("B", self.beat_info)
            self.conn.send(self.getInfo())
            self.last_time = time.time()

    @loger.PathwayLoger
    def pathway(self, stu): 
        '''
        @brief: 控制轨道:0xAA不处理轨道信息 0x00(弹丸在正中央) 0x01(弹丸在右侧) -0x01(弹丸在左侧)
        '''
        self.setID("chassis")
        self.setDATA("b", stu)
        self.conn.send(self.getInfo())
        return stu

    # @loger.PathwayLoger
    def devError(self, stu): 
        '''
        @brief: 控制轨道:0xAA不处理轨道信息 0x00(弹丸在正中央) 0x01(弹丸在右侧) -0x01(弹丸在左侧)
        '''
        self.setID("deverror")
        self.setDATA("b", stu)
        self.conn.send(self.getInfo())
        return stu

class Sentry_up(Robot):
    def __init__(self, conn, name = "sentry_up"):
        super().__init__(conn, name)

class Hero(Robot):
    def __init__(self, conn, name="hero"):
        super().__init__(conn, name)

class Sentry_down(Robot):
    def __init__(self, conn, name = "sentry_down"):
        super().__init__(conn, name)

class Infantry(Robot):
    def __init__(self, conn, name = "infantry"):
        super().__init__(conn, name)

class Engineer(Robot):
    def __init__(self, conn, name = "engineer"):
        super().__init__(conn, name)