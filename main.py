from ultralytics import YOLO

datapath = r'C:\Users\Admin\Desktop\CatDog'
model = YOLO('yolov8n-cls.pt')
result = model.train(data = datapath,epochs = 500,imgsz = 640)