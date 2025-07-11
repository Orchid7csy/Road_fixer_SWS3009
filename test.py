from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Load a pretrained YOLO11n model
model = YOLO("/home/csy/My_baby_blue/Road_fixer/runs/segment/train/weights/best.pt")

# Define path to directory containing images and videos for inference
source = "/home/csy/My_baby_blue/Road_fixer/nus_testset"

output_dir = "./test_results"
os.makedirs(output_dir, exist_ok=True)

# 最简单的预处理方案
def quick_fix(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    
    # 遮蔽检测到的直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (128, 128, 128), 8)
    
    return image
s

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

# Visualize the results
for j, r in enumerate(results):
    # 获取原始文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename("./nus_testset"))[0]
    
    output_filename = f"{base_name}_{j}.jpg"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # 使用r.save保存结果，指定完整路径
    r.save(filename=output_path)