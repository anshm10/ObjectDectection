import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

os.makedirs("dataset", exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)

count=0
while count < 10:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Capture", color_image)

    # Save Image
    cv2.imwrite(f"dataset/image_{count}.jpg", color_image)
    print(f"Saved dataset/image_{count}.jpg")
    
    time.sleep(3)

    count = count + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()