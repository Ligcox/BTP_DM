'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-07-21 16:26:50
Description: 
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.client import device_lib

import cv2
import time
import numpy as np

# TF_MODEL = r'/home/nvidia/Desktop/BTPDM/src/saved_model'
TF_MODEL = r'C:\Users\1\Desktop\Competition\2021全国大学生机器人大赛RoboMaster对抗赛\BTP&DM\src\saved_model'
class NumClassifier(object):
    def __init__(self):
        print("{:=^40}".format(""))
        print("{:=^40}".format("Number classifier launching..."))
        print("{:=^40}".format(""))
        print(device_lib.list_local_devices())
        print(tf.__version__)

        print("Loding model:", TF_MODEL)
        self.model = tf.keras.models.load_model(TF_MODEL)
        print("Number classifier launch success")
        print("Model details:")
        print(self.model.summary())

    def process(self, image):
        ImageCrop = cv2.resize(image, (640,480))
        ImageCrop = cv2.resize(image, (64,64))

        # 去除红蓝色
        frame = cv2.cvtColor(ImageCrop, cv2.COLOR_BGR2HSV)
        final_mask = np.ones(frame.shape[:2], np.uint8)*255
        _, low_mask = cv2.threshold(
            frame[:, :, 1], 100, 255, cv2.THRESH_BINARY)
        _, high_mask = cv2.threshold(
            frame[:, :, 1],255, 255, cv2.THRESH_TOZERO_INV)
        channel_masks = cv2.bitwise_and(low_mask, high_mask)
        final_mask = cv2.bitwise_and(final_mask, channel_masks)
        output = cv2.bitwise_and(frame, frame, mask=final_mask)
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        final_mask = cv2.subtract(ImageCrop, output)

        cv2.imshow("a", final_mask)
        cv2.waitKey(1)
        
        num = self.getImgClass(final_mask)
        return 0

    def getImgClass(self, img):
        """
        数字分类
        """
        img = tf.keras.preprocessing.image.array_to_img(img)
        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)
        pred_class = self.model(img, training = False)
        pred_class2 = list(pred_class)
        tmp_list = sorted(pred_class2[0])

        if tmp_list[len(tmp_list) - 1] - tmp_list[len(tmp_list) - 2] > 1.3:
            num = np.argmax(pred_class) + 1
            print("class", num)
            return num
        else:
            return 0

if __name__ == '__main__':
    c = NumClassifier()
    while True:
        # img = cv2.imread(r'/home/nvidia/Desktop/BTPDM/src/Samples/classifierSample.jpg')
        img = cv2.imread(r'C:\Users\1\Desktop\classifierSample.jpg')
        st = time.time()
        c.process(img)
        print(time.time()-st)