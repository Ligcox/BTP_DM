import cv2
from collections import OrderedDict
import config.globalVarManager as gvm
import numpy as np
#############################
######## 全局参数设定 ########
#############################
CATEGOTY_DEFAULT_TYPE = gvm.get_value("CATEGOTY_DEFAULT_TYPE")
DEBUG_MODEL = gvm.get_value("DEBUG_MODEL")

#########################
######## 参数设定 ########
#########################

######## TrackBar控制选项 ########
# 数据类型为字典类 (dict类)
# 数据格式为
# {
#   控制项名称	:   [类型，默认数据值，辅助配置]
# }
#
# 目前共有两种有效控制类
# 1. NUMERIC - 数值类
# 2. ENUM    - 枚举类 (ENUMERATE太长)
# 其中
#
# NUMERIC 的 辅助配置格式为
# [最小值，最大值，精度] (均为控制数值对应值，Trackbar相关数值交给程序计算)
#
# ENUM 的 辅助配置格式为 字典类，其中key暂时不用，value为对应数据
#
# TODO：
# 字典类排序比较混乱，key不用的话可以考虑换list
# 如果需要用key的话也许需要其他办法
#


# 定义Control类数值位置常量
IDX_CONTROL_TYPE = 0
IDX_CONTROL_VAL = 1
IDX_CONTROL_SETTING = 2

# 定义Control类型
TYPE_CONTROL_NONE = 0  # 占位，方便处理报错
TYPE_CONTROL_NUMERIC = 1
TYPE_CONTROL_ENUM = 2

# 定义Trackbar类数值位置常量
IDX_NUMERIC_MINVAL = 0
IDX_NUMERIC_MAXVAL = 1
IDX_NUMERIC_RES = 2

# 辅助BOOLEAN类实现的常量MAPPING
BOOLEAN_ENUM = {
    'False': False,
    'True': True
}
# 定义装甲板信息数据辅助信息
ROI_RECT = 0
ROI_BOX = 1
PNP_INFO = 2
ROI_DATA_LENGTH = 3
ROI_RECT_LENGTH = 2
#  定义灯带数据信息
Energy_Strip = 0
Circle_Strip = 1
Strip_DataLength = 2

# 定义能量机关点辅助信息
Energy_Strip_Point = 0
Circle_Center_Point = 1
Point_DataLength = 2
#  定义顺时针,逆时针
clockwise = 0
anticlockwise = 1
#  定义能量机关预测模式

# 视频文件控制
video_controls = {
    'alpha'	:	[TYPE_CONTROL_NUMERIC, 1, [0, 10, 0.1]],
    'beta'		 :	[TYPE_CONTROL_NUMERIC, -133, [-255, 255, 1]],
    'rotation':   [TYPE_CONTROL_NUMERIC, 0, [-180, 180, 90]],
    'scale':   [TYPE_CONTROL_NUMERIC, 0.5, [0, 1, 0.1]],
    'silent':   [TYPE_CONTROL_ENUM, 0, BOOLEAN_ENUM]
}


######## TrackBar辅助项 ########
# TrackBar读取数字值，套一层Mapping换成枚举值
# 摄像头编码
camera_codecs = {
    'V4L2_PIX_FMT_YUYV':   1448695129,
    # Not supported, 这个编码我们用的摄像头不支持
    # 'V4L2_PIX_FMT_H264' :   875967048,
    'V4L2_PIX_FMT_MJPEG':   1196444237
}

# 摄像头控制集
camera_controls = {
    'FourCC':   [TYPE_CONTROL_ENUM, 0, camera_codecs],
    'Brightness':   [TYPE_CONTROL_NUMERIC, 64, [0, 128, 1]],
    'Contrast':   [TYPE_CONTROL_NUMERIC, 32, [0, 95, 1]],
    'Saturation':   [TYPE_CONTROL_NUMERIC, 32, [0, 100, 1]],
    'Hue':   [TYPE_CONTROL_NUMERIC, 3000, [0, 4000, 100]],
    'Gain':   [TYPE_CONTROL_NUMERIC, 0, [0, 80, 1]],
    'Exposure':   [TYPE_CONTROL_NUMERIC, 20000, [0, 40000, 1000]],
    'Gamma':   [TYPE_CONTROL_NUMERIC, 150, [100, 300, 1]],
    'Sharpness':   [TYPE_CONTROL_NUMERIC, 2, [1, 7, 1]],
    'rotation':   [TYPE_CONTROL_NUMERIC, 0, [-180, 180, 90]],
    'scale':   [TYPE_CONTROL_NUMERIC, 0.4, [0, 5, 0.1]],
    'silent':   [TYPE_CONTROL_ENUM, 0, BOOLEAN_ENUM]
}

daheng_cam_controls = {
    'rotation':   [TYPE_CONTROL_NUMERIC, 0, [-180, 180, 90]],
    'scale':   [TYPE_CONTROL_NUMERIC, 0.7, [0, 5, 0.1]],
    'silent':   [TYPE_CONTROL_ENUM, 0, BOOLEAN_ENUM],
    'Contrast': [TYPE_CONTROL_NUMERIC, -50, [-50, 100, 10]],
    'Gamma': [TYPE_CONTROL_NUMERIC, 0.1, [0.1, 10, 0.1]],
    'ExposureTime': [TYPE_CONTROL_NUMERIC, 1000, [20, 1000000, 1000]],
    'saturation': [TYPE_CONTROL_NUMERIC, 64, [0, 128, 4]],
    'sharpen': [TYPE_CONTROL_NUMERIC, 1, [0.1, 5, 0.2]]

}

# 灯条捕捉Mask类型的Mapping，分为Binary和to_zero两种，
# Binary    满足条件的全为真，反之为假 -> 表现为黑白图像 真=255 | 假=0 -》 即消除亮度信息，做Mask时大部分情况是消除的，因为原图含有亮度信息
# To_zero   满足条件的不变，反之归零 -> 表现为黑白图像 不变=0~255 | 归零=0
channelMasks = {
    'Binary':   [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV],
    'Normal':   [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
}

# 比赛类型设置   0为红方，1为蓝方
categoryDefault = {
    0: [20, 20, 240, 15, 240, 20],
    1: [90, 8, 210, 20, 235, 20]
}
# 能量机关比赛类型设置   0为击打红方，1为击打蓝方
categoryEnergyDefault = {
    0: [0, 36, 0, 255, 47, 255],
    1: [80, 120, 204, 255, 114, 255]
}
# Filter控制类集
channel_filter_controls = {
    # 使用RGB空间时使用以下配置
    'blueTarget'	:	[TYPE_CONTROL_NUMERIC, 245, [0, 255, 1]],
    'blueTolerance'	:	[TYPE_CONTROL_NUMERIC, 20, [0, 20, 1]],
    'greenTarget'	:	[TYPE_CONTROL_NUMERIC, 245, [0, 255, 1]],
    'greenTolerance':	[TYPE_CONTROL_NUMERIC, 20, [0, 20, 1]],
    'redTarget'		:	[TYPE_CONTROL_NUMERIC, 245, [0, 255, 1]],
    'redTolerance'	:	[TYPE_CONTROL_NUMERIC, 10, [0, 20, 1]],

    # 使用HSV空间时使用以下配置
    'hueTarget'	:			[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][0], [0, 255, 1]],
    'hueTolerance'	:		[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][1], [0, 20, 1]],
    'saturationTarget'	:	[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][2], [0, 255, 1]],
    'saturationTolerance':	[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][3], [0, 20, 1]],
    'valueTarget'		:	[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][4], [0, 255, 1]],
    'valueTolerance'	:	[TYPE_CONTROL_NUMERIC, categoryDefault[CATEGOTY_DEFAULT_TYPE][5], [0, 100, 1]],

    'maskMethod':   [TYPE_CONTROL_ENUM, 0, channelMasks],
    'silent':   [TYPE_CONTROL_ENUM, 0, BOOLEAN_ENUM]
}

# 能量机关Filter控制类集
energy_filter_controls = {

    "threshold": [TYPE_CONTROL_NUMERIC, 30, [0, 250, 1]],
    'lowHue': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][0], [0, 255, 1]],
    'highHue': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][1], [0, 255, 1]],
    'lowSat': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][2], [0, 255, 1]],
    'highSat': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][3], [0, 255, 1]],
    'lowVal': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][4], [0, 255, 1]],
    'highVal': [TYPE_CONTROL_NUMERIC, categoryEnergyDefault[CATEGOTY_DEFAULT_TYPE][5], [0, 255, 1]],
    "aug": [TYPE_CONTROL_NUMERIC, 0, [0, 20, 1]],
    "iterations": [TYPE_CONTROL_NUMERIC, 1, [0, 20, 1]]

}


# 颜色Mapping，方便调整标记的颜色
color_map = {
    # Key -> Color in RGB Code '#FF0000':   (0, 0, 255),
    '# F00':   (0, 255, 0),
    '# 0FF':   (255, 0, 0),
    '# F00':   (0, 255, 255),
    '# 0FF':   (255, 0, 255),
    '# FFF':   (255, 255, 0)
}

# 装甲板等条控制类
# bigArmour：大装甲板（2021赛季为230*127）
# normalArmour:小装甲板（2021赛季为135*125）
# 注意：下面得参数为灯条尺寸，非装甲板尺寸
armourSize = {
    "bigArmour": [235, 58],
    "normalArmour": [140, 58],
    "energyArmour": [230, 127]
}

# Detector控制集
simple_detector_controls = {
    # 轮廓线筛选，0为不设限
    'Contour_Area_Threshold':   [TYPE_CONTROL_NUMERIC, 0.001, [0, 0.01, 0.001]],
    # 长边最大夹角，0为不设限
    'LightStrip_Angle_Diff':   [TYPE_CONTROL_NUMERIC, 3, [0, 90, 1]],
    # 间距与较长一方长边比的最小值，0为不设限
    'LightStrip_Displacement_Diff_Min':   [TYPE_CONTROL_NUMERIC, 1, [0, 5, 0.1]],
    'LightStrip_Displacement_Diff_Max':   [TYPE_CONTROL_NUMERIC, 2.8, [0, 5, 0.1]],
    'LightStrip_Box_Color':   [TYPE_CONTROL_ENUM, 1, color_map],
    'LightStrip_Box_Thickness':   [TYPE_CONTROL_NUMERIC, 2, [1, 10, 1]],
    'Armor_Box_Color':   [TYPE_CONTROL_ENUM, 2, color_map],
    'Armor_Box_Thickness':   [TYPE_CONTROL_NUMERIC, 2, [1, 10, 1]],
    'silent':   [TYPE_CONTROL_ENUM, 0, BOOLEAN_ENUM],
    "bigArmour":    [TYPE_CONTROL_ENUM, 0, armourSize],
    "normalArmour": [TYPE_CONTROL_ENUM, 1, armourSize],
    "energyArmour": [TYPE_CONTROL_ENUM, 2, armourSize]
}

# 能量机关Detector控制集
energy_detector_controls = {
    "area_lower_value_energyStrip": [TYPE_CONTROL_NUMERIC, 1200, [0, 10000, 1]],
    "area_upper_value_energyStrip": [TYPE_CONTROL_NUMERIC, 6800, [0, 10000, 1]],

    "area_lower_value_energyTar": [TYPE_CONTROL_NUMERIC, 10, [0, 10000, 1]],
    "area_upper_value_energyTar": [TYPE_CONTROL_NUMERIC, 3000, [0, 10000, 1]],

    "area_lower_value_circleStrip": [TYPE_CONTROL_NUMERIC, 100, [0, 1000, 1]],
    "area_upper_value_circleStrip": [TYPE_CONTROL_NUMERIC, 800, [0, 6000, 1]],
    "Dilate_value": [TYPE_CONTROL_NUMERIC, 25, [0, 40, 1]],
    "iterations": [TYPE_CONTROL_NUMERIC, 1, [0, 40, 1]],
    "energyArmour": [TYPE_CONTROL_ENUM, 2, armourSize]
}
# 能量机关Predictor控制集
energy_predictor_controls = {
    "lead_time": [TYPE_CONTROL_NUMERIC, 0.4, [0, 1, 0.01]],
    "lead_angle": [TYPE_CONTROL_NUMERIC, 10, [0, 40, 1]],
    "energyArmour": [TYPE_CONTROL_ENUM, 2, armourSize]
                             }

# decision控制集
decision_controls = {
    'yaw_angle_offset':   [TYPE_CONTROL_NUMERIC, 2.60, [-5, 5, 0.01]],
    'pitch_angle_offset':   [TYPE_CONTROL_NUMERIC, 1.9, [-5, 5, 0.01]],
}

# 哨兵下云台decision控制集
sentryUp_decision_controls = {
    'yaw_angle_offset':   [TYPE_CONTROL_NUMERIC, 3.85, [-10, 10, 0.01]],
    'pitch_angle_offset':   [TYPE_CONTROL_NUMERIC, 0, [-10, 10, 0.01]],
    "pitch_pnp_error":   [TYPE_CONTROL_NUMERIC, 0.28, [0, 0.5, 0.01]]
}

# 哨兵下云台decision控制集
sentryDown_decision_controls = {
    'yaw_angle_offset':   [TYPE_CONTROL_NUMERIC, 3.85, [-10, 10, 0.01]],
    'pitch_angle_offset':   [TYPE_CONTROL_NUMERIC, 0, [-10, 10, 0.01]],
    "pitch_pnp_error":   [TYPE_CONTROL_NUMERIC, 0.28, [0, 0.5, 0.01]]
}

hero_decision_controls = {
    'yaw_angle_offset':   [TYPE_CONTROL_NUMERIC, 4.61, [-10, 10, 0.01]],
    'pitch_angle_offset':   [TYPE_CONTROL_NUMERIC, 2.25, [-10, 10, 0.01]],
    'BarrelPtzOffSetY': [TYPE_CONTROL_NUMERIC, -100, [-200, 200, 10]]

}

infantry_decision_controls = {
    'yaw_angle_offset':   [TYPE_CONTROL_NUMERIC, 2.25, [-10, 10, 0.01]],
    'pitch_angle_offset':   [TYPE_CONTROL_NUMERIC, 2.93, [-10, 10, 0.01]],
    'BarrelPtzOffSetY': [TYPE_CONTROL_NUMERIC, -100, [-200, 200, 10]]
}


# Realsense控制集
auto_dodge_controls = {
'realsense_middle': [TYPE_CONTROL_NUMERIC, 100, [0, 848, 1]]
}

predict_controls = {
"A" : np.array([
            [1, 0, 0.0333, 0],
            [0, 1, 0, 0.0333],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),

"B" : np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),

"H" : np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),

"Q" : np.array([
            [0.01, 0, 0, 0],
            [0, 0.01, 0, 0],
            [0, 0, 0.01, 0],
            [0, 0, 0, 0.01]
        ]),

"R" : np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1]
        ]),

"N" : np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]),

"initial_x" : np.array([0, 0, 0, 0]),

"initial_p" : np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])
}