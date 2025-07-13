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

后处理逻辑：
1. 先用YOLO识别出所有裂缝和地砖缝
2. 基于YOLO识别结果创建掩码
3. 在掩码区域内使用HoughLines检测直线
4. 在原YOLO结果图上绘制检测到的直线
'''

def postprocess_with_hough(original_image, yolo_results):
    """
    后处理函数：在YOLO检测结果的基础上使用HoughLines检测直线
    
    Args:
        original_image: 原始图像
        yolo_results: YOLO检测结果
    
    Returns:
        processed_image: 叠加了直线检测结果的图像
    """
    # 获取YOLO结果图像
    result_image = yolo_results.plot()  # BGR格式
    
    # 创建掩码用于限制直线检测区域
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    # 如果有检测框，在掩码上标记这些区域
    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        for box in yolo_results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # 如果有分割掩码，使用分割结果
    if hasattr(yolo_results, 'masks') and yolo_results.masks is not None:
        for mask_data in yolo_results.masks.data:
            # 将mask转换为numpy数组并调整尺寸
            mask_np = mask_data.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, mask_binary)
    
    # 如果没有检测到任何目标，使用整个图像
    if np.sum(mask) == 0:
        mask = np.ones(original_image.shape[:2], dtype=np.uint8) * 255
    
    # 在掩码区域内进行边缘检测和直线检测
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 应用掩码
    masked_gray = cv2.bitwise_and(gray, mask)
    
    # Canny边缘检测
    edges = cv2.Canny(masked_gray, 50, 150)
    
    # HoughLinesP直线检测
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,      # 降低阈值以检测更多直线
        minLineLength=30,  # 降低最小长度
        maxLineGap=20      # 增加最大间隙
    )
    
    # 在YOLO结果图上绘制检测到的直线
    if lines is not None:
        print(f"检测到 {len(lines)} 条直线")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 用红色绘制直线，线宽为2
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("未检测到直线")
    
    return result_image

# 初始化YOLO模型
model = YOLO('./runs/segment/train/weights/best.pt')  # 请根据您的模型路径调整

# 设置文件夹路径
input_folder = './nus_testset'
output_folder = './test_results_postprocess'

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
        
        # 先在原始图像上运行YOLO检测
        results = model(original_image)
        
        # 处理结果
        for j, r in enumerate(results):
            print(f"  检测到 {len(r.boxes) if r.boxes is not None else 0} 个目标")
            
            # 进行后处理：在YOLO结果基础上检测直线
            processed_image = postprocess_with_hough(original_image, r)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_postprocess_result_{j}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # 保存结果到指定文件夹
            cv2.imwrite(output_path, processed_image)
            print(f"  结果已保存到: {output_path}")
        
        print(f"完成处理: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")

print(f"\n所有图像处理完成！结果已保存到 {output_folder} 文件夹中。")
print("后处理说明：")
print("- 红色线条表示在YOLO检测结果基础上识别的直线")
print("- 直线检测仅在YOLO识别的裂缝和地砖缝区域内进行")
print("- 结果图像同时显示YOLO检测结果和直线检测结果")