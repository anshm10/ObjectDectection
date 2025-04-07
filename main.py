import pyrealsense2 as rs
import numpy as np
import cv2

# check if the camera is available
context = rs.context() # gives access to connected devices and other RealSense functionalities.
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline() # pipeline object is used to configure and start stream
cfg = rs.config() # config object is used to specifiy what streams you want

# TODO: try with a usbc to usbc cable instead of usbb
#                            max: 1920, 1080                 30
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# supported resolutions according to the datasheet:
# 1280 x 720    6, 15, 30
# 848 x 480     6, 15, 30, 60, 90
# 640 x 480     6, 15, 30, 60, 90
# 640 x 360     6, 15, 30, 60, 90
# 480 x 270     6, 15, 30, 60, 90
# 424 x 240     6, 15, 30, 60, 90
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipe.start(cfg) # initialize streams from camera

while True:
    frame = pipe.wait_for_frames()

    depth = frame.get_depth_frame()
    color = frame.get_color_frame()

    # convert both frames to numpy arrays for manipulation
    depth_image = np.asanyarray(depth.get_data())
    smoothed_depth = cv2.medianBlur(depth_image, 5)
    color_image = np.asanyarray(color.get_data())

    # point_x, point_y = 250, 100
    # distance_mm = depth_image[point_y, point_x]
    # cv2.circle(color_image, (point_x, point_y), 8, (0, 0, 255), -1)
    # cv2.putText(color_image, "{} mm".format(distance_mm), (point_x, point_y - 10), 0, 1, (0, 0, 255), 2) 

    # save depth numpy array to file (for milestone)
    with open('depthdata.txt', 'w') as f:
        np.savetxt(f, depth_image, fmt='%d')

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(smoothed_depth, alpha=0.5), cv2.COLORMAP_JET)

    cv2.imshow("depth", depth_colormap)
    cv2.imshow("color", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

pipe.stop()