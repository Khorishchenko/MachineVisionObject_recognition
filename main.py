

# https://youtu.be/Uj4O2_dwRiA?si=qzZvSsMFyZTeCMm7
# https://itproger.com/ua/news/raspoznavanie-obaektov-na-python-glubokoe-mashinnoe-obuchenie
# https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection
# https://imageai.readthedocs.io/en/latest/video/index.html


# imageai and tensorflow

# version 1.0 ---  image


# from imageai.Detection import ObjectDetection
# import os

# exec_path = os.getcwd()

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath(os.path.join(
# 	exec_path, "resnet50_coco_best_v2.0.1.h5")
# )
# detector.loadModel()

# list = detector.detectObjectsFromImage(
# 	input_image=os.path.join(exec_path, "objects.jpg"),
# 	output_image_path=os.path.join(exec_path, "new_objects.jpg"),
# 	minimum_percentage_probability=90,
# 	display_percentage_probability=True,
# 	display_object_name=False
# )


# version 1.1 --- Video
import cv2
import numpy as np
from imageai.Detection import VideoObjectDetection
import os

import tkinter as tk


# print(os.path.exists("coco.names"))
# print(os.path.exists("yolov3.cfg")) 
# print(os.path.exists("yolov3.cfg"))

# root = tk.Tk()
# root.title("Tkinter Test")
# root.geometry("200x100")

# label = tk.Label(root, text="Tkinter працює!")
# print("Tkinter працює!")

# label.pack()

# root.mainloop()



execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()
# detector.useCPU()


video_path = detector.detectObjectsFromVideo(
	input_file_path=os.path.join(execution_path, "medium_small.mp4"),
	output_file_path=os.path.join("/Users/admin/Desktop", "detected"),
	frames_per_second=20,
	log_progress=True
)


print(video_path)