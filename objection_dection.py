import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline() 
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

model = YOLO("runs/detect/train7/weights/best.pt")

def detect_objects(frame):
    """Run YOLO object detection."""
    results = model(frame, conf=0.8)
    return results

# def detect_lines(gray_frame):
#     """Apply edge detection to find drawn lines."""
#     edges = cv2.Canny(gray_frame, 50, 150)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
#     return lines

def main():
    while True:
        frame = pipe.wait_for_frames()

        depth = frame.get_depth_frame()
        color = frame.get_color_frame()

        depth_image = np.asanyarray(depth.get_data()) 
        color_image = np.asanyarray(color.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # Object Detection
        results = detect_objects(color_image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = boxes_conf[i]
                cls = int(boxes_cls[i])
                label = model.names[cls]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                cv2.putText(color_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # # Line Detection
        # lines = detect_lines(gray_image)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)


        cv2.imshow("Depth Detection", depth_colormap)
        cv2.imshow('Object Detection', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()