if __name__ == '__main__':    
    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')

    results = model.train(data='C:\\Users\\valid\\Desktop\\data\\config.yaml', epochs=50, imgsz=640 , workers=1, batch=6 )

    