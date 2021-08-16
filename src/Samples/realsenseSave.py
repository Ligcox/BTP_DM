'''
Author: Ligcox
Date: 2021-07-27 19:41:18
LastEditors: Ligcox
LastEditTime: 2021-07-27 22:50:37
Description: 
Apache License  (http://www.apache.org/licenses/)
Shanghai University Of Engineering Science
Copyright (c) 2021 Birdiebot R&D department
'''
import os
import cv2
import numpy as np
import time
import tqdm

if os.name == "nt": 
    import pyrealsense2 as rs
elif os.name == "posix":
    import pyrealsense2.pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 0, 848, 100, rs.format.z16, 300)
config.enable_record_to_file("0727testdata.bag")
pipeline.start(config)

# Create colorizer object
colorizer = rs.colorizer()

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
num = 5000

for i in tqdm.tqdm(range(num)):
    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    # Get depth frame
    depth_frame = frames.get_depth_frame()

    # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # Render image in opencv window
    cv2.imshow("Depth Stream", depth_color_image)
    key = cv2.waitKey(1)
    # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
pipeline.stop()