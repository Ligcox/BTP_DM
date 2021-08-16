from collections import OrderedDict


# ######################################通讯协议相关######################################
# ################################通讯协议按照下列格式发送#################################
#
# 帧头    目标地址    功能码  数据长度    数据内容    和校验    附加校验
# HEAD    D_ADDR       ID      LEN       DATA     SUMCHECK  ADDCHECK
'''
# 帧头
# 固定为0xFF

# 目标地址
# 广播地址：0x00
# 上位机：0x01
# 机器人：0x02-0x08

# 功能码
# 轨道：0x01
# 云台yaw：0x02
# 云台pitch：0x03
# 枪管：0x04
# 裁判系统信息0x05

# 数据长度
# 根据需要计算

# 数据内容
# 轨道:0x00(空缺) 0x00轨道静止 0x01加速 0x02减速
# 云台偏移量: -655~655
# 枪管: UINT8 初速度设置   BOOL   是否发射


大小端转换参考：
https://docs.python.org/3/library/struct.html
https://www.cnblogs.com/coser/archive/2011/12/17/2291160.html
'''

O_INFO = OrderedDict()
O_INFO["HEAD"] = 0xFF
O_INFO["D_ADDR"] = None
O_INFO["ID"] = None
O_INFO["LEN"] = None
O_INFO["DATA"] = bytearray()
O_INFO["SUM_CHECK"] = None
O_INFO["ADD_CHECK"] = None
D_INFO = O_INFO

D_ADDR = {
    "broadcast": 0x00,
    "mainfold": 0x01,
    "sentry_up": 0x02,
    "sentry_down": 0x03,
    "infantry": 0x04,
    "engineer": 0x05,
    "hero": 0x06,
    "air": 0x07,
    "radar": 0x08
}

ID = {
    # 下位机地址
    "chassis": 0x01,

    # 云台数据相关
    "gimbal": 0x02,
    "gimbal_angle":0x21,

    "barrel": 0x04,
    "mode": 0x06,

    # 上位机模式控制
    "manifold_ctrl":0x50,
    "referee_system": 0x51,

    # 心跳数据
    "heartbeat":0xAA,

    # 设备故障重新插拔设备
    "deverror":0x70
}

LEN = {
    "LEN": None
}


DATA = {
    "pathway": bytearray([0x00, 0x00]),
    "sentry_up": bytearray([0x00, 0x00]),
    "sentry_down": bytearray([0x00, 0x00]),
    "barrel": bytearray([0x00, 0x00]),
}


def sumcheck_cal(INFO):
    sumcheck = 0
    addcheck = 0
    for i in [(k, v) for k, v in INFO.items()][:-3]:
        sumcheck += i[1]
        addcheck += sumcheck
    
    for i in INFO["DATA"]:
        sumcheck += i
        addcheck += sumcheck

    SUM_CHECK = int(sumcheck) & 0XFF
    ADD_CHECK = int(addcheck) & 0XFF
    return SUM_CHECK, ADD_CHECK

# 下位机状态
STATUS = {
    "mode": 0,
    "isShoot": 0,
    "pathway_direction": 0,
    "pathway_speed":0,
    "yaw_angle":0,
    "pitch_angle":0
}