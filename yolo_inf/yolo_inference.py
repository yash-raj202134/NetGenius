from ultralytics import YOLO 

model = YOLO('models/yolov8x')

model.predict("yolo_inf/image.png",save = True)

