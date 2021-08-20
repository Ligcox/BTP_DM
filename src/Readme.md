<!--
 * @Author: Ligcox
 * @Date: 2021-08-20 01:04:37
 * @LastEditors: Ligcox
 * @LastEditTime: 2021-08-20 01:06:13
 * @Description: Readme file
 * Apache License  (http://www.apache.org/licenses/)
 * Shanghai University Of Engineering Science
 * Copyright (c) 2021 Birdiebot R&D department
-->
BTPDM程序源文件
===

# 代码目录说明
## 框架主体功能部分
    - main.py：框架程序主入口
    - module.py：主要功能模块基类
    - scheduler.py：任务调度器，通过BCP数据实现在不同任务间切换
    - decision.py：框架决策层，对图像信息处理生成的数据转化为机器人的决策信息转换位BCP数据发送至下位机
    - utils.py：全局辅助函数设置及文件第三方库导入
    - vInput.py：图像读取模块，根据配置文件
    - targetDetector.py：功能模块，目标识别器，包括了灯条识别、装甲板识别、能量机关识别等
    - imageFilter.py：功能模块，色彩过滤器，将图像中关键信息提取并进行腐蚀膨胀等处理

## config:**配置相关文件**
    - config/config.py：主程序任务相关配置
    - config/connConfig.py：木鸢通讯协议相关配置
    - config/devConfig.py：设备、图像输入源、线程等待时间相关配置
    - config/globalVarManager.py：全局变量管理器，全局变量相关配置

## Sample:**演示样例文件**
    - c2py/：展示了C++和python进行混合编程的例子
    - performanceAnalysis/：展示了BTPDM的性能分析及BTPDM的性能分析日志
    - BCPSample.ipynb：展示了如何对数据解析成BCP数据帧
    - classifierDemo.py：展示了使用已经完成训练的模型进行装甲板数字识别推理