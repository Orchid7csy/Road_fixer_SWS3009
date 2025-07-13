#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys
import os

# 添加核心模块路径 - 使用更robust的导入方式
try:
    # 尝试相对导入
    from ..core.depth_graph_node import DepthGraphNode
except ImportError:
    # 如果相对导入失败，使用绝对路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from depth_graph_node import DepthGraphNode

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')
        
        # 创建发布者
        self.depth_publisher = self.create_publisher(
            Image, 
            '/depth/image_raw', 
            10
        )
        
        # 创建订阅者 - 订阅原始图像
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # CV桥接器
        self.cv_bridge = CvBridge()
        
        # 初始化深度处理模块
        self.depth_processor = DepthGraphNode()
        
        # 定时器 - 用于定期处理（如果需要）
        self.timer = self.create_timer(0.1, self.process_depth)
        
        self.get_logger().info('Depth Publisher Node initialized')
        
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 使用你的深度处理模块
            depth_map = self.depth_processor.process_image(cv_image)
            
            # 发布深度图
            self.publish_depth(depth_map, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def publish_depth(self, depth_map, header):
        """发布深度图像"""
        try:
            # 转换深度图为ROS消息
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map, "32FC1")
            depth_msg.header = header
            depth_msg.header.frame_id = "depth_frame"
            
            # 发布
            self.depth_publisher.publish(depth_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing depth: {str(e)}')
    
    def process_depth(self):
        """定期处理函数（可选）"""
        pass

def main(args=None):
    rclpy.init(args=args)
    
    node = DepthPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()