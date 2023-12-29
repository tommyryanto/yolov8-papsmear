# from ultralytics import YOLO
# import cv2
#
# model = YOLO('../YOLOV8-PAPSMEAR/Yolo-Weights/best.pt')
# results = model("../YOLOV8-PAPSMEAR/Images/L-CV 1904662-140007 (1).jpg",show=True)
# cv2.waitKey(0)

from ultralytics import YOLO
import cv2
import os
from skimage.io import imread

# define the location of the folder to go through
directory = 'Images/'

# get a list of files in that folder
file_list = os.listdir(directory)
image_file_list = [file for file in file_list if file.endswith(".jpg")]
model = YOLO('../pythonProject/Yolo-Weights/best.pt')
for image_file in image_file_list:
    image = model(imread(directory + image_file), show=True)
    # results = model("Images/L-CV 1904662-140007 (1).jpg",show=True)
    cv2.waitKey(0)
