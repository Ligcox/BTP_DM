'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-20 01:18:46
Description: Program main entry, by initializing different types of robot scheduler to perform different tasks.
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''

import config.globalVarManager as gvm
# 比赛全局类型设置
gvm.set_value("CATEGOTY_DEFAULT_TYPE", "category", "red")
gvm.set_value("DEBUG_MODEL", "DEBUG_SETTING", True)
from scheduler import *

# 启动任务调度器, 选择合适的调度器实现机器人初始化
# scheduler = SentryUpScheduler
# scheduler = SentryDownScheduler
# scheduler = HeroScheduler
scheduler = InfantryScheduler
# scheduler = RadarScheduler

# 通过任务调度器开启任务
with scheduler() as sch:
    sch.run()