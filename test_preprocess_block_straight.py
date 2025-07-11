import cv2
import numpy as np
from PIL import Image
import os
import glob
from ultralytics import YOLO

'''
HoughLinesP 直线检测参数:

threshold: 投票阈值,越高检测到的直线越少
minLineLength: 最小直线长度,越大只检测长直线
maxLineGap: 直线段间最大间隙,影响直线连接
rho=1: 距离分辨率(像素精度)
theta=np.pi/180: 角度分辨率(1度精度)

'''

# 最简单的预处理方案
def quick_fix(image):
    # 创建图像副本，不改变原图
    processed_image = image.copy()
    
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
    
    # 遮蔽检测到的直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(processed_image, (x1, y1), (x2, y2), (100, 100, 100), 4)
    
    return processed_image

# 初始化YOLO模型
model = YOLO('./runs/segment/train/weights/best.pt')  # 请根据您的模型路径调整

# 设置文件夹路径
input_folder = './nus_testset'
output_folder = './test_results'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 支持的图像格式
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

# 获取所有图像文件
image_files = []
for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, extension)))
    image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

print(f"找到 {len(image_files)} 张图像")

# 处理每张图像
for i, image_path in enumerate(image_files):
    print(f"处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
    
    try:
        # 读取原始图像
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 应用预处理（不改变原图）
        preprocessed_image = quick_fix(original_image)
        
        # 在预处理后的图像上运行YOLO检测
        results = model(preprocessed_image)
        
        # 处理结果
        for j, r in enumerate(results):
            # 绘制结果
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_preprocessed_result_{j}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # 保存结果到指定文件夹
            cv2.imwrite(output_path, im_bgr)
            print(f"结果已保存到: {output_path}")
        
        print(f"完成处理: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")

print(f"\n所有图像处理完成！结果已保存到 {output_folder} 文件夹中。")