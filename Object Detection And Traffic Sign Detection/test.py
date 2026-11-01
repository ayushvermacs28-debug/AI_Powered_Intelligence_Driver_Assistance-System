from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
results = model("input/road.png")
results = model("input/road2.png")