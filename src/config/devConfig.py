'''
Author: Ligcox
Date: 2021-04-19 20:32:54
LastEditors: Ligcox
LastEditTime: 2021-07-29 02:17:55
Description: 
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
'''
Author: your name
Date: 2021-04-19 20:32:54
LastEditTime: 2021-07-24 13:54:11
LastEditors: Ligcox
Description: In User Settings Edit
FilePath: \BTP&DM\src\config\devConfig.py
'''
######################################################################
#######################摄像头参数设置###################`###############
######################################################################
# source = 文件名以使用视频文件
# source = r"D:\PycharmProjects\BTP_DM\2blue.mp4"
# source = 数字ID以使用摄像头
# source = DAHENG+数字ID以使用摄像头
# source = 0
source = "DAHENG0"

######################################################################
#######################串口参数设置###################################
######################################################################
# # 串口号
PORTX = "COM7"
# # 波特率
BPS = 500000
# 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
TIMEX = 0

import platform
sysstr = platform.system()
if(sysstr =="Windows"):
    PORTX = "COM1"
elif(sysstr == "Linux"):
    PORTX = "/dev/ttyTHS0"
    source = "DAHENG0"
else:
    raise KeyError("run the pro in windows or linux")


######################################################################
######################线程执行时间设置#################################
######################################################################

threading_time = {
    # UART发送函数等待时间
    "tx_threading": 0.002,
    # UART接收函数等待时间
    "rx_threading": 0.001,
    # 图像读取函数等待时间
    "imread_threading": 0,
    # 心跳函数等待时间
    "heartbeat_threading": 0.045,
    # 主函数等待时间
    "scheduler_threading": 0
}

def getThreadingSleepTime(name):
    return threading_time[name]