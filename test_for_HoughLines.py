import cv2
import numpy as np
import os
import glob

'''
HoughLinesP 直线检测参数:
threshold: 投票阈值,越高检测到的直线越少
minLineLength: 最小直线长度,越大只检测长直线
maxLineGap: 直线段间最大间隙,影响直线连接
rho=1: 距离分辨率(像素精度)
theta=np.pi/180: 角度分辨率(1度精度)
'''

# 在图像上绘制检测到的直线
def draw_lines_on_image(image):
    """
    在输入图像上检测直线并用红线绘制出来。
    """
    # 创建图像副本，不改变原图
    processed_image = image.copy()
    
    # 转换为灰度图并进行边缘检测
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
    
    # 在图像上用红色绘制检测到的直线
    if lines is not None:
        print(f"检测到 {len(lines)} 条直线。")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 使用红色 (0, 0, 255) 和粗细为 2 的线条绘制
            cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    return processed_image

# --- 主程序 ---

# 设置输入文件夹路径
input_folder = './nus_testset'  # 请确保这个文件夹存在

# 支持的图像格式
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

# 获取所有图像文件
image_files = []
for extension in image_extensions:
    # 忽略大小写，合并搜索结果
    image_files.extend(glob.glob(os.path.join(input_folder, extension.lower())))
    image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

# 检查是否找到图像
if not image_files:
    print(f"在文件夹 '{input_folder}' 中未找到任何支持的图像文件。")
else:
    print(f"找到 {len(image_files)} 张图像。")

# 处理每张图像
for i, image_path in enumerate(image_files):
    print(f"\n--- 正在处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
    
    try:
        # 读取原始图像
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"无法读取图像，跳过: {image_path}")
            continue
            
        # 在图像上绘制直线检测结果
        image_with_lines = draw_lines_on_image(original_image)
        
        # 显示结果图像
        cv2.imshow(f"Line Detection Result for {os.path.basename(image_path)}", image_with_lines)
        
        # 等待用户按键，然后继续处理下一张或退出
        print("按任意键继续处理下一张图像...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"处理图像 {image_path} 时发生错误: {str(e)}")

print("\n所有图像处理完成！")