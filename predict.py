from ultralytics import YOLO

model = YOLO("C:\\Users\\valid\\Desktop\\data\\runs\\detect\\train\\weights\\last.pt")

model.predict('safty.mov', save=True , show=True , conf=0.7)
# model.predict('', save=True , show=True , conf=0.7 , save_text = True)

# Gradio is an open-source Python library that is used to build machine learning and data science demos and web applications.