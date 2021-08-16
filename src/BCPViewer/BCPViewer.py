import matplotlib.pyplot as plt
import numpy as np
import threading
import sys
from random import random, randrange
import time
import os


# 退出标志位
QUIT_FLAG = False


class DATA():
    def __init__(self, name=None):
        self.name = name
        self.x = []
        self.y = []

    def setData(self, x, y):
        if len(self.x)>1000:
            self.x = self.x[1000:]
        if len(self.y)>1000:
            self.y = self.y[1000:]
        self.x.append(x)
        self.y.append(y)

    def getData(self):
        return [self.x, self.y]


class BCPViewer(object):
    """BCPViewer主界面
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 3)
        self.sub_title = ["yaw", "pitch", "distance", "Kalman_yaw",
                          "Kalman_pitch", "Kalman_distance", "FPS", "test", "test", "test"]
        self.fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 设置子图之间的间距
        self.fig.canvas.set_window_title("BCPViewer")  # 设置窗口标题

        # 子图字典，key为子图的序号，value为子图句柄
        self.axdict = {
            0: self.ax[0, 0],
            1: self.ax[0, 1],
            2: self.ax[0, 2],
            3: self.ax[1, 0],
            4: self.ax[1, 1],
            5: self.ax[1, 2],
            6: self.ax[2, 0],
            7: self.ax[2, 1],
            8: self.ax[2, 2]
        }
        self.init_data_node()
        self.init_ax()

        self.monitor_t = time.time()

    def init_data_node(self):
        self.yaw_node = DATA("yaw")
        self.pitch_node = DATA("pitch")
        self.distance_node = DATA("distance")
        self.Kalman_yaw_node = DATA("Kalman_yaw")
        self.Kalman_pitch_node = DATA("Kalman_pitch")
        self.Kalman_distance_node = DATA("Kalman_distance")
        self.FPS_node = DATA("test0_node")
        self.test1_node = DATA("test1_node")
        self.test2_node = DATA("test2_node")

    def init_ax(self):
        self.axdict[0].set_ylim(-10, 10)
        self.axdict[1].set_ylim(-10, 10)
        self.axdict[4].set_ylim(-10, 10)
        self.axdict[5].set_ylim(-10, 10)
        self.axdict[6].set_ylim(0, 100)

    def showPlot(self):
        """ 显示曲线 """
        plt.show()

    def setPlotStyle(self, index):
        """ 设置子图的样式，这里仅设置了标题 """
        self.axdict[index].set_title(self.sub_title[index], fontsize=12)

    def update(self, index, x, y):
        """
        更新指定序号的子图
        :param index: 子图序号
        :param x: 横轴数据
        :param y: 纵轴数据
        :return:
        """
        self.setPlotStyle(index)  # 设置子图样式
        self.axdict[index].plot(x, y, "bo", markersize= 1)  # 绘制最新的数据
        if time.time()-self.monitor_t > 10:
            self.axdict[index].set_xlim(
                time.time()-10, time.time())  # 根据X轴数据区间调整X轴范围
        plt.draw()


def Main(bcpViewer):
    """
    模拟收到实时数据，更新曲线的操作
    :param plot: 曲线实例
    :return:
    """
    node_list = [bcpViewer.yaw_node, bcpViewer.pitch_node, bcpViewer.distance_node, bcpViewer.Kalman_yaw_node,
                 bcpViewer.Kalman_pitch_node, bcpViewer.Kalman_distance_node, bcpViewer.FPS_node]
    global QUIT_FLAG
    log = open(os.path.abspath(os.path.join(os.path.split(
        os.path.realpath(__file__))[0], "../log/log.log")))
    while True:
        if QUIT_FLAG:
            break
        info = log.readline()
        while info != "":
            info = info.split(",")
            name, datatime, y = info
            name, datatime, y = name, float(datatime), float(y)
            # print(info)
            if name == "T0":
                bcpViewer.yaw_node.setData(datatime, y)
            if name == "T1":
                bcpViewer.pitch_node.setData(datatime, y)
            if name == "T2":
                bcpViewer.distance_node.setData(datatime, y)
            if name == "T3":
                bcpViewer.Kalman_yaw_node.setData(datatime, y)
            if name == "T4":
                bcpViewer.Kalman_pitch_node.setData(datatime, y)
            if name == "T5":
                bcpViewer.Kalman_distance_node.setData(datatime, y)
            if name == "T6":
                bcpViewer.FPS_node.setData(datatime, y)
                
            info = log.readline()
        for i in range(7):
            bcpViewer.update(i, *node_list[i].getData())
        time.sleep(1)


bcpViewer = BCPViewer()

main = threading.Thread(target=Main, args=(bcpViewer,))  # 启动一个线程更新曲线数据
main.start()

bcpViewer.showPlot()  # showPlot方法会阻塞当前线程，直到窗口关闭
print("BCPViewer close")
QUIT_FLAG = True  # 通知更新曲线数据的线程退出
main.join()
