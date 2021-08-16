from config import *
from utils import *
from config.config import *

# Place Holder,占位用图像，以免silent模式的时候Trackbar被截掉
BASE_GRAYSCALE = 240
BLANK_IMG = np.ones([1, 400], np.uint8) * BASE_GRAYSCALE

# 基础模块
class module(object):
    # 初始化部分元素，在这里初始化的部分为独有部分，这样__init__可以最大化被子模块继承
    controls = None
    name = "Empty module"
    # class通用初始化，子模块可以重载后通过super().__init__()调用该初始化功能

    def __init__(self, hide_controls=True):
        self.hide_controls = hide_controls
        self.control_window_name = self.name + "Control"
        if DEBUG_MODEL:
            self.createControlWindow()
    # 响应Trackbar控制，key在建立Trackbar时通过partial给出，实际仅接受trackbar读取值value，转换后存至self.controls

    def updateControls(self, key, trackbar_value):
        try:
            self.setControlVal(key, trackbar_value)
        except Exception as e:
            print(e.args)
    # 建立控制窗口

    def createControlWindow(self):
        if not self.hide_controls:
            if not self.controls is None:
                try:
                    cv2.namedWindow(self.control_window_name,
                                    cv2.WINDOW_AUTOSIZE)
                    for key in self.controls:
                        cv2.createTrackbar(key, self.control_window_name, self.getControlTrackBarPos(
                            key), self.getControlTrackbarMax(key), partial(self.updateControls, key))
                except:
                    pass

    # 预留处理接口
    def process(self, input):
        return None

    # with语句入口
    def __enter__(self):
        return self

    # with语句出口
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    # 结束任务时的清理程序
    def cleanup(self):
        try:
            cv2.destroyWindow(self.control_window_name)
        except:
            pass

    # Controls相关辅助function
    def getControlType(self, key):
        val = TYPE_CONTROL_NONE
        try:
            val = self.controls[key][IDX_CONTROL_TYPE]
        except:
            pass
        return val

    # 获取当前Control的详细数据
    def getControlData(self, key):
        val = None
        try:
            val = self.controls[key][IDX_CONTROL_SETTING]
        except:
            pass
        return val

    # 仅作为参考，实际使用相关数据时仅用一次类别判断，不然太浪费
    def getControlMin(self, key):
        val = 0
        try:
            t = self.getControlType(key)
            if t == TYPE_CONTROL_NUMERIC:
                val = self.getControlData(key)[IDX_NUMERIC_MINVAL]
        except:
            pass
        return val

    def getControlMax(self, key):
        val = 0
        try:
            t = self.getControlType(key)
            if t == TYPE_CONTROL_NUMERIC:
                val = self.getControlData(key)[IDX_NUMERIC_MAXVAL]
            elif t == TYPE_CONTROL_ENUM:
                val = len(self.getControlData(key)) - 1
        except:
            pass
        return val

    def getControlRes(self, key):
        val = 1
        try:
            t = self.getControlType(key)
            if t == TYPE_CONTROL_NUMERIC:
                val = self.getControlData(key)[IDX_NUMERIC_RES]
        except:
            pass
        return val

    # 获取Trackbar最小值设定,因为OpenCV限制，这个值一直是0
    def getControlTrackbarMin(self, key):
        return 0

    # 获取Trackbar最大值设定
    def getControlTrackbarMax(self, key):
        val = 0
        try:
            t = self.getControlType(key)
            d = self.getControlData(key)
            if t == TYPE_CONTROL_ENUM:
                val = len(d) - 1
            elif t == TYPE_CONTROL_NUMERIC:
                val = int((d[IDX_NUMERIC_MAXVAL]-d[IDX_NUMERIC_MINVAL])
                          * 1.0 / d[IDX_NUMERIC_RES])
        except:
            pass
        return val

    # 获取当前Control值对应的Trackbar位置
    def getControlTrackBarPos(self, key):
        val = 0
        try:
            t = self.getControlType(key)
            d = self.getControlData(key)
            val = self.controls[key][IDX_CONTROL_VAL]
            if t == TYPE_CONTROL_NUMERIC:
                val = int((val - d[IDX_NUMERIC_MINVAL])
                          * 1.0 / d[IDX_NUMERIC_RES])
        except:
            val = 0
        return val

    # 转换Trackbar读取值至Control控制值
    def setControlVal(self, key, trackbar_value):
        try:
            t = self.getControlType(key)
            d = self.getControlData(key)
            if t == TYPE_CONTROL_ENUM:
                self.controls[key][IDX_CONTROL_VAL] = trackbar_value  # 跳过验证和转换
            elif t == TYPE_CONTROL_NUMERIC:
                self.controls[key][IDX_CONTROL_VAL] = (
                    trackbar_value * d[IDX_NUMERIC_RES]) + d[IDX_NUMERIC_MINVAL]
        except:
            pass

    # 获取Control控制值(如果是枚举类型则在这里转换)
    def getControlVal(self, key):
        val = None
        try:
            t = self.getControlType(key)
            d = self.getControlData(key)
            val = self.controls[key][IDX_CONTROL_VAL]
            if t == TYPE_CONTROL_ENUM:
                val = list(d.values())[val]
        except:
            pass
        return val
