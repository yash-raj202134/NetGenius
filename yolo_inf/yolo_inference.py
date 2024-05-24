from ultralytics import YOLO 

model = YOLO('models/last.pt')

result = model.predict("yolo_inf/input_video.mp4",conf = 0.2,save = True)
print(result)

for box in result[0].boxes:
    print(box)

