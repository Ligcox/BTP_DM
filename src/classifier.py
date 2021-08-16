'''
Author: Ligcox
Date: 2021-04-06 15:20:21
LastEditors: Ligcox
LastEditTime: 2021-08-10 15:22:07
Description: The principal implementation of the classifier
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import tensorflow as tf
import copy
from utils import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image


class NumClassifier(object):
    def __init__(self):
        self.model = tf.keras.models.load_model(r'src/saved_model')
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print(tf.__version__)

        # print(self.model.summary())

    def process(self, image, armour_list):
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
        final = cv2.subtract(ImageCrop, output)
        
        if len(armour_list) != 0:
            for armour in armour_list:
                ROI_RECT, ROI_BOX, PNP_LIST = armour
                boxes = ROI_BOX
                dis = PNP_LIST[0]
                ImageCrop = self.cropImage(final, boxes, dis)
                num = self.getImgClass(ImageCrop)
                return num
        else:
            return 0

    def cropImage(self, image, boxes, dis):
        """
        图像截取，截取出数字部分
        """
        width, height = 64, 64  # 截取图片的宽和高
        boxes = boxes.tolist()
        a = 25 - int(dis / 250)
        b = 30 - int(dis / 250)
        boxes = SortCoordinate(boxes)  # 坐标排序
        boxes = ExpandRegion(boxes, (a, b))  # 截取区域增大

        matrix = TransformPerspective(boxes, (64, 64))  # 透视变换
        ImageCrop = cv2.warpPerspective(image, matrix, (width, height))  # 转换为图片
        ImageCrop = aug(ImageCrop)
        cv2.imshow("ImageCrop",ImageCrop)
        return ImageCrop

    def getImgClass(self, img):
        """
        数字分类
        """
        try:
            # img_cpd = copy.deepcopy(img)
            # img = cv2.resize(img, (64, 64))
            img = tf.keras.preprocessing.image.array_to_img(img)
            img = image.img_to_array(img)

            img = np.expand_dims(img, axis=0)
            pred_class = self.model(img)
            pred_class2 = list(pred_class)
            tmp_list = sorted(pred_class2[0])

            if tmp_list[len(tmp_list) - 1] - tmp_list[len(tmp_list) - 2] > 1.3:
                num = np.argmax(pred_class) + 1
                print(num)
                return num
            else:
                return 0
        except Exception as e:
            print(e)
