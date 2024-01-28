from ultralytics import YOLO

datapath = r'place path to your root folder'
model = YOLO('yolov8n-cls.pt')
result = model.train(data = datapath,epochs = 500,imgsz = 640)
