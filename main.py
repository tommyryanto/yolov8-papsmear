from ultralytics import YOLO
import cv2

model = YOLO('../YOLOV8-PAPSMEAR/Yolo-Weights/best.pt')
results = model("../YOLOV8-PAPSMEAR/Images/L-CV 1904662-140007 (1).jpg",show=True)
cv2.waitKey(0)
