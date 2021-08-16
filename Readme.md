<!--
 * @Author: Ligcox
 * @Date: 2021-04-06 15:20:21
 * @LastEditors: Ligcox
 * @LastEditTime: 2021-08-16 16:02:15
 * @Description: 
 * Apache License  (http://www.apache.org/licenses/)
 * Shanghai University Of Engineering Science
 * Copyright (c) 2021 Birdiebot R&D department
-->
BTP&DM
===
Birdiebot目标感知与决策框架
---
BTP&DM是一个针对RMUC/RMUL/RMUT开发的一款开源框架，你可以通过BTP&DM快速构建机器人。该项目受到上海工程技术大学上海市大学生创新创业基地木鸢机甲工作室的资助，参与该项目请联系zyhbum@foxmail.com

BTP&DM包含了一下内容：
- BCP（Birdiebot Communication Protocol，birdiebot通讯协议）
- 基于opencv的传统视觉目标特征检测，包含装甲板识别及能量机关识别
- 对于赛场环境的决策处理
- BCPViewer对使用BCP数据的实时UI显示
- 使用深度学习完成目标识别与分类

编码风格指南
---
BTP&DM推荐使用PEP8编码规范编写的python代码，在开始修改BTP&DM前，请谨记：**A Foolish Consistency is the Hobgoblin of Little Minds**  
在编辑和维护BTP&DM前，请保证代码可读性和规范
- 养成良好的命名规范，正确对变量名进行命名
- 正确的引入需要的packages
- 请确保所有模块、重要功能模块添加了注释，请使用_Strunk and White, The Elements of Style_中推荐的注释风格
- 对异常进行检查并抛出相应的异常信息，任何意外出现的异常都可能直接导致失去一局比赛

详细的编码风格请参阅[PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

版权和许可信息
---
BTP&DM使用 Apache Licence 2.0许可证，鼓励在尊重木鸢机甲工作室及上海工程技术大学的著作权基础上的代码共享，欢迎各参赛队参与项目修改与维护，并作为开源或商业软件再次发布。  
但注意
- 您需要在修改的代码中保留Apache Licence
- 如果你使用了修改了代码，需要在被修改的文件中说明。
- 在延伸的代码中（修改和有源代码衍生的代码中）需要带有原来代码中的协议，商标，专利声明和其他BTP&DM、木鸢机甲工作室及上海工程技术大学所规定需要包含的说明。
- 如果再发布的产品中包含一个Notice文件，则在Notice文件中需要带有Apache Licence。你可以在Notice中增加自己的许可，但不可以表现为对Apache Licence构成更改。
木鸢机甲工作对其发行的或与合作伙伴共同开发的BTP&DM，受各国版权法及国际版权公约的保护。  
对于上述版权内容，超越合理使用范畴、并未经木鸢机甲工作室书面许可的使用行为，木鸢机甲工作室均保留追究法律责任的权利。

相关法律
---
- 《中华人民共和国刑法》(节录)
- 《中华人民共和国商标法》
- 《全国人民代表大会常务委员会关于维护互联网安全的决定》
- 《计算机信息网络国际联网安全保护管理办法》
- 《计算机软件保护条例》


[Birdiebot目标感知与决策框架](https://github.com/Ligcox/BTP_DM/wiki/Birdiebot%E7%9B%AE%E6%A0%87%E6%84%9F%E7%9F%A5%E4%B8%8E%E5%86%B3%E7%AD%96%E6%A1%86%E6%9E%B6)
[BTPDM优化方向和Roadmap](https://github.com/Ligcox/BTP_DM/wiki/BTPDM%E4%BC%98%E5%8C%96%E6%96%B9%E5%90%91%E5%92%8CRoadmap)
[功能模块：BCPloger及BCPViewer](https://github.com/Ligcox/BTP_DM/wiki/%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97%EF%BC%9ABCPloger%E5%8F%8ABCPViewer)
[功能模块：任务调度器](https://github.com/Ligcox/BTP_DM/wiki/%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97%EF%BC%9A%E4%BB%BB%E5%8A%A1%E8%B0%83%E5%BA%A6%E5%99%A8)
[功能模块：木鸢通讯协议](https://github.com/Ligcox/BTP_DM/wiki/%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97%EF%BC%9A%E6%9C%A8%E9%B8%A2%E9%80%9A%E8%AE%AF%E5%8D%8F%E8%AE%AE)
[功能模块：能量机关识别及击打任务](https://github.com/Ligcox/BTP_DM/wiki/%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97%EF%BC%9A%E8%83%BD%E9%87%8F%E6%9C%BA%E5%85%B3%E8%AF%86%E5%88%AB%E5%8F%8A%E5%87%BB%E6%89%93%E4%BB%BB%E5%8A%A1)
[功能模块：装甲板识别及自瞄任务](https://github.com/Ligcox/BTP_DM/wiki/%E5%8A%9F%E8%83%BD%E6%A8%A1%E5%9D%97%EF%BC%9A%E8%A3%85%E7%94%B2%E6%9D%BF%E8%AF%86%E5%88%AB%E5%8F%8A%E8%87%AA%E7%9E%84%E4%BB%BB%E5%8A%A1)
[性能分析](https://github.com/Ligcox/BTP_DM/wiki/%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90)
[文件目录结构及文件用途说明](https://github.com/Ligcox/BTP_DM/wiki/%E6%96%87%E4%BB%B6%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84%E5%8F%8A%E6%96%87%E4%BB%B6%E7%94%A8%E9%80%94%E8%AF%B4%E6%98%8E)
[木鸢通讯协议概述](https://github.com/Ligcox/BTP_DM/wiki/%E6%9C%A8%E9%B8%A2%E9%80%9A%E8%AE%AF%E5%8D%8F%E8%AE%AE%E6%A6%82%E8%BF%B0)
[测试分析及效果展示](https://github.com/Ligcox/BTP_DM/wiki/%E6%B5%8B%E8%AF%95%E5%88%86%E6%9E%90%E5%8F%8A%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA)
[算法分析：位姿解算](https://github.com/Ligcox/BTP_DM/wiki/%E7%AE%97%E6%B3%95%E5%88%86%E6%9E%90%EF%BC%9A%E4%BD%8D%E5%A7%BF%E8%A7%A3%E7%AE%97)
[系统框架概述](https://github.com/Ligcox/BTP_DM/wiki/%E7%B3%BB%E7%BB%9F%E6%A1%86%E6%9E%B6%E6%A6%82%E8%BF%B0)
[设计思想与设计模式](https://github.com/Ligcox/BTP_DM/wiki/%E8%AE%BE%E8%AE%A1%E6%80%9D%E6%83%B3%E4%B8%8E%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F)