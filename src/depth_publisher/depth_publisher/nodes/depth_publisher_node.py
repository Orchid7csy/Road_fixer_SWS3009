#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys
import os
import torch
from PIL import Image as PILImage
import depth_pro

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_publisher')
        
        # 创建发布者 - 发布深度图像
        self.depth_publisher = self.create_publisher(
            Image, 
            '/depth/image_raw', 
            10
        )
        
        # 创建发布者 - 发布相机信息（包含焦距信息）
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/depth/camera_info',
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
        
        # 初始化 depth-pro 模型
        self.init_depth_pro()
        
        self.get_logger().info('Depth Publisher Node initialized with depth-pro')
        
    def init_depth_pro(self):
        """初始化 depth-pro 模型"""
        try:
            # 加载模型和预处理转换
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model.eval()
            
            # 检查是否有GPU可用
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            self.get_logger().info(f'Depth-pro model loaded on {self.device}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize depth-pro: {str(e)}')
            raise
    
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 转换为RGB格式（depth-pro需要RGB）
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_image = PILImage.fromarray(rgb_image)
            
            # 使用 depth-pro 处理图像
            depth_map, focal_length_px = self.process_with_depth_pro(pil_image)
            
            # 发布深度图
            self.publish_depth(depth_map, msg.header)
            
            # 发布相机信息
            self.publish_camera_info(focal_length_px, msg.header, cv_image.shape)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_with_depth_pro(self, pil_image):
        """使用 depth-pro 处理图像"""
        try:
            # 获取图像尺寸用于焦距计算
            width, height = pil_image.size
            
            # 由于我们没有真实的焦距，使用 None 让模型估算
            # 如果你有真实的焦距信息，可以在这里提供
            f_px = None
            
            # 预处理图像
            image_tensor = self.transform(pil_image)
            
            # 添加batch维度并移动到设备
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 运行推理
            with torch.no_grad():
                prediction = self.model.infer(image_tensor, f_px=f_px)
            
            # 获取深度和焦距
            depth = prediction["depth"]  # 深度以米为单位
            focallength_px = prediction["focallength_px"]  # 焦距以像素为单位
            
            # 转换为numpy数组
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            
            # 确保深度图是正确的形状和数据类型
            if depth.ndim == 3:
                depth = depth.squeeze()
            
            # 转换为32位浮点数
            depth = depth.astype(np.float32)
            
            return depth, focallength_px
            
        except Exception as e:
            self.get_logger().error(f'Error in depth-pro processing: {str(e)}')
            raise
    
    def publish_depth(self, depth_map, header):
        """发布深度图像"""
        try:
            # 转换深度图为ROS消息
            # 使用 32FC1 格式表示单通道32位浮点深度图
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_map, "32FC1")
            
            # 设置头信息
            depth_msg.header = header
            depth_msg.header.frame_id = "depth_frame"
            
            # 发布
            self.depth_publisher.publish(depth_msg)
            
            self.get_logger().debug(f'Published depth image with shape: {depth_map.shape}')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing depth: {str(e)}')
    
    def publish_camera_info(self, focal_length_px, header, image_shape):
        """发布相机信息"""
        try:
            camera_info_msg = CameraInfo()
            
            # 设置头信息
            camera_info_msg.header = header
            camera_info_msg.header.frame_id = "depth_frame"
            
            # 设置图像尺寸
            height, width = image_shape[:2]
            camera_info_msg.height = height
            camera_info_msg.width = width
            
            # 设置相机内参矩阵
            # K = [fx  0  cx]
            #     [ 0 fy  cy]
            #     [ 0  0   1]
            fx = fy = float(focal_length_px) if focal_length_px is not None else 525.0
            cx = width / 2.0
            cy = height / 2.0
            
            camera_info_msg.k = [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
            ]
            
            # 设置投影矩阵
            # P = [fx  0  cx  0]
            #     [ 0 fy  cy  0]
            #     [ 0  0   1  0]
            camera_info_msg.p = [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            # 设置畸变模型（假设无畸变）
            camera_info_msg.distortion_model = "plumb_bob"
            camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # 设置rectification矩阵为单位矩阵
            camera_info_msg.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
            
            # 发布相机信息
            self.camera_info_publisher.publish(camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing camera info: {str(e)}')

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