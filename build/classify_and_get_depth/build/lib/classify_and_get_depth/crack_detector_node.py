#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Polygon, PolygonStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys
import os

# 添加核心模块路径 - 使用更robust的导入方式
try:
    # 尝试相对导入
    from ..core.ft_yolov11 import YoloV11CrackDetector
    from ..core.filter_straight_lines import FilterStraightLines
except ImportError:
    # 如果相对导入失败，使用绝对路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from src.classify_and_get_depth.classify_and_get_depth.ft_yolov11 import YoloV11CrackDetector
    from filter_straight_lines import FilterStraightLines

class CrackDetector(Node):
    def __init__(self):
        super().__init__('crack_detector')
        
        # 创建订阅者
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.depth_subscriber = self.create_subscription(
            Image,
            '/depth/image_raw',
            self.depth_callback,
            10
        )
        
        # 创建发布者
        self.crack_publisher = self.create_publisher(
            PolygonStamped,
            '/crack_detection/polygons',
            10
        )
        
        self.debug_image_publisher = self.create_publisher(
            Image,
            '/crack_detection/debug_image',
            10
        )
        
        # CV桥接器
        self.cv_bridge = CvBridge()
        
        # 初始化检测器
        self.crack_detector = YoloV11CrackDetector()
        self.line_filter = FilterStraightLines()
        
        # 存储最新的深度图
        self.latest_depth = None
        
        self.get_logger().info('Crack Detector Node initialized')
        
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 裂缝检测
            results = self.crack_detector.detect(cv_image)
            
            # 过滤直线
            filtered_results = self.line_filter.filter(results)
            
            # 发布检测结果
            self.publish_crack_polygons(filtered_results, msg.header)
            
            # 发布调试图像
            debug_image = self.draw_debug_image(cv_image, filtered_results)
            self.publish_debug_image(debug_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def depth_callback(self, msg):
        """处理接收到的深度图消息"""
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {str(e)}')
    
    def publish_crack_polygons(self, crack_results, header):
        """发布裂缝多边形"""
        try:
            for crack in crack_results:
                polygon_msg = PolygonStamped()
                polygon_msg.header = header
                polygon_msg.header.frame_id = "camera_frame"
                
                # 构建多边形点
                for point in crack['polygon']:
                    p = Point()
                    p.x = float(point[0])
                    p.y = float(point[1])
                    
                    # 如果有深度信息，添加z坐标
                    if self.latest_depth is not None:
                        p.z = float(self.latest_depth[int(point[1]), int(point[0])])
                    else:
                        p.z = 0.0
                    
                    polygon_msg.polygon.points.append(p)
                
                self.crack_publisher.publish(polygon_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error publishing polygons: {str(e)}')
    
    def draw_debug_image(self, image, results):
        """绘制调试图像"""
        debug_img = image.copy()
        
        for crack in results:
            # 绘制检测框
            if 'bbox' in crack:
                x1, y1, x2, y2 = crack['bbox']
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制多边形
            if 'polygon' in crack:
                pts = np.array(crack['polygon'], np.int32)
                cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
        
        return debug_img
    
    def publish_debug_image(self, image, header):
        """发布调试图像"""
        try:
            debug_msg = self.cv_bridge.cv2_to_imgmsg(image, "bgr8")
            debug_msg.header = header
            self.debug_image_publisher.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing debug image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    node = CrackDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()