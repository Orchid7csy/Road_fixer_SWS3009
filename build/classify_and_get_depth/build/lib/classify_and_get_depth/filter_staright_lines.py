from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import math
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import os

def extract_skeleton_features(mask):
    """
    提取骨架特征来判断是否为直线
    """
    # 确保mask是二值图像
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # 获取骨架
    skeleton = skeletonize(mask)
    
    if np.sum(skeleton) == 0:
        return False, {}
    
    # 获取骨架点
    skeleton_points = np.column_stack(np.where(skeleton))
    
    if len(skeleton_points) < 10:
        return False, {}
    
    # 计算特征
    features = {}
    
    # 1. 计算骨架的端点数量
    endpoints = find_endpoints(skeleton)
    features['num_endpoints'] = len(endpoints)
    
    # 2. 计算骨架的分支点数量
    branch_points = find_branch_points(skeleton)
    features['num_branches'] = len(branch_points)
    
    # 3. 计算骨架的总长度
    features['skeleton_length'] = np.sum(skeleton)
    
    # 4. 计算骨架的直线度
    features['straightness'] = calculate_straightness(skeleton_points)
    
    # 5. 计算骨架的弯曲度
    features['curvature'] = calculate_curvature(skeleton_points)
    
    # 判断是否为直线（地砖边缘）
    is_line = (
        features['straightness'] > 0.95 and  # 直线度很高
        features['curvature'] < 0.02 and     # 弯曲度很低
        features['num_branches'] == 0        # 没有分支
    )
    
    return not is_line, features

def find_endpoints(skeleton):
    """
    找到骨架的端点
    """
    # 3x3卷积核，用于计算邻域
    kernel = np.ones((3, 3), np.uint8)
    
    # 计算每个点的邻域数量
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # 端点：自身为1且邻域数量为2（包括自身）
    endpoints = np.where((skeleton == 1) & (neighbors == 2))
    
    return list(zip(endpoints[0], endpoints[1]))

def find_branch_points(skeleton):
    """
    找到骨架的分支点
    """
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # 分支点：自身为1且邻域数量>3
    branch_points = np.where((skeleton == 1) & (neighbors > 3))
    
    return list(zip(branch_points[0], branch_points[1]))

def calculate_straightness(points):
    """
    计算点序列的直线度
    """
    if len(points) < 3:
        return 0
    
    # 计算最远两点间的直线距离
    distances = cdist(points, points)
    max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
    
    start_point = points[max_dist_idx[0]]
    end_point = points[max_dist_idx[1]]
    
    straight_distance = np.linalg.norm(end_point - start_point)
    
    # 计算所有点的路径长度
    path_length = len(points)  # 简化计算
    
    if path_length == 0:
        return 0
    
    # 直线度 = 直线距离 / 路径长度
    straightness = straight_distance / path_length
    
    return straightness

def calculate_curvature(points):
    """
    计算路径的平均曲率
    """
    if len(points) < 3:
        return 0
    
    # 对点进行排序，形成连续路径
    sorted_points = sort_points_by_path(points)
    
    curvatures = []
    for i in range(1, len(sorted_points) - 1):
        # 计算相邻三点的角度
        p1, p2, p3 = sorted_points[i-1], sorted_points[i], sorted_points[i+1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算角度
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0

def sort_points_by_path(points):
    """
    将点按路径顺序排序
    """
    if len(points) < 2:
        return points
    
    # 使用简单的最近邻排序
    sorted_points = [points[0]]
    remaining = list(points[1:])
    
    while remaining:
        current = sorted_points[-1]
        distances = [np.linalg.norm(current - p) for p in remaining]
        nearest_idx = np.argmin(distances)
        sorted_points.append(remaining[nearest_idx])
        remaining.pop(nearest_idx)
    
    return np.array(sorted_points)

def hough_line_filter(mask, line_threshold=0.7):
    """
    使用霍夫变换检测直线
    """
    # 边缘检测
    edges = cv2.Canny(mask, 50, 150)
    
    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                           minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return True  # 没有检测到直线，可能是裂缝
    
    # 计算直线覆盖的面积
    line_mask = np.zeros_like(mask)
    total_line_length = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
        total_line_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # 计算直线覆盖率
    line_coverage = np.sum(line_mask > 0) / np.sum(mask > 0)
    
    # 如果直线覆盖率太高，认为是地砖边缘
    return line_coverage < line_threshold

def geometric_filter(mask, aspect_ratio_threshold=15, compactness_threshold=0.2):
    """
    基于几何特征过滤
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # 取最大轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 计算几何特征
    area = cv2.contourArea(main_contour)
    if area < 50:
        return False
    
    # 最小外接矩形
    rect = cv2.minAreaRect(main_contour)
    w, h = rect[1]
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    
    # 周长
    perimeter = cv2.arcLength(main_contour, True)
    
    # 紧凑度
    compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # 判断是否为直线特征
    is_line = (aspect_ratio > aspect_ratio_threshold and 
               compactness < compactness_threshold)
    
    return not is_line

def combined_filter(mask, method='all'):
    """
    组合过滤器
    """
    if np.sum(mask) == 0:
        return False
    
    results = []
    
    if method in ['skeleton', 'all']:
        try:
            is_crack, _ = extract_skeleton_features(mask)
            results.append(is_crack)
        except:
            pass
    
    if method in ['hough', 'all']:
        try:
            is_crack = hough_line_filter(mask)
            results.append(is_crack)
        except:
            pass
    
    if method in ['geometric', 'all']:
        try:
            is_crack = geometric_filter(mask)
            results.append(is_crack)
        except:
            pass
    
    if not results:
        return True  # 如果所有方法都失败，保守地认为是裂缝
    
    # 投票机制
    crack_votes = sum(results)
    return crack_votes > len(results) / 2

def process_yolo_with_advanced_filter(model_path, source_path, output_dir="advanced_results"):
    """
    使用高级过滤器处理YOLO结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    results = model(source_path, stream=True)
    
    for i, r in enumerate(results):
        original_img = r.orig_img
        
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            
            # 创建结果图像
            result_img = original_img.copy()
            crack_count = 0
            
            for mask in masks:
                # 调整mask大小
                mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # 应用组合过滤器
                if combined_filter(mask_binary, method='all'):
                    crack_count += 1
                    
                    # 绘制裂缝
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
                    
                    # 添加标签
                    if contours:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(result_img, f"Crack {crack_count}", (cx, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"filtered_result_{i}.jpg")
            cv2.imwrite(output_path, result_img)
            print(f"处理完成: {output_path}, 检测到 {crack_count} 个裂缝")

# 简化版本，适合快速测试
def simple_line_filter(mask):
    """
    简化的直线过滤器
    """
    # 使用霍夫变换检测直线
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                           minLineLength=30, maxLineGap=5)
    
    if lines is None:
        return True  # 没有直线，可能是裂缝
    
    # 如果检测到很多直线，可能是地砖边缘
    return len(lines) < 3

# 使用示例
if __name__ == "__main__":

    model_path = "/home/csy/My_baby_blue/Road_fixer/runs/segment/train/weights/best.pt"
    source_path = "/home/csy/My_baby_blue/Road_fixer/nus_testset"
    
    # 处理结果
    process_yolo_with_advanced_filter(model_path, source_path)
    
    print("处理完成！请查看 advanced_results 文件夹中的结果。")