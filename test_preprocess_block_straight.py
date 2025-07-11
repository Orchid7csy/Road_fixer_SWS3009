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

# 删除 quick_fix 函数定义及其调用，并替换为后处理函数
def draw_lines_on_image(base_img, ref_img, line_color=(0, 0, 255), thickness=2):
    """
    在 base_img 上根据 ref_img 的边缘检测结果绘制霍夫直线。

    参数说明:
        base_img: 需要绘制结果的图像(BGR)
        ref_img: 参与霍夫直线检测的参考图像(BGR)，一般为原始图像
        line_color: 绘制直线的颜色，默认为红色
        thickness: 直线粗细
    返回:
        带直线绘制的图像
    """
    # 保证不修改原始输入
    output = base_img.copy()

    # Canny 边缘检测
    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 霍夫直线检测 (P 版本更易于得到线段两端点)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)

    # 在 base_img 上绘制直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), line_color, thickness)

    return output

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
        
        # 在原图像上运行YOLO检测
        results = model(original_image)

        # 处理结果
        for j, r in enumerate(results):
            # 获得 YOLO 绘制的结果图像 (包含分割可视化)
            im_bgr = r.plot()  # BGR-order numpy array

            # 在结果图像上绘制霍夫直线（使用原始图像进行直线检测）
            im_bgr_with_lines = draw_lines_on_image(im_bgr, original_image)

            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_with_lines_{j}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # 保存带有直线的检测结果
            cv2.imwrite(output_path, im_bgr_with_lines)
            print(f"结果已保存到: {output_path}")
        
        print(f"完成处理: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")

print(f"\n所有图像处理完成！结果已保存到 {output_folder} 文件夹中。")