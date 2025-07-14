#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys
import os
from typing import List, Dict, Optional
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformListener, Buffer
import message_filters
from threading import Lock

# 导入motion planning模块
try:
    # 尝试相对导入
    from motion_planning import DepthBasedMotionPlanner, VehicleState, DefectInfo
except ImportError:
    # 如果相对导入失败，使用绝对路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from motion_planning import DepthBasedMotionPlanner, VehicleState, DefectInfo

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planning_node')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('vehicle_width', 1.8),
                ('vehicle_length', 4.5),
                ('planning_rate', 10.0),
                ('max_planning_distance', 20.0),
                ('camera_frame', 'camera_link'),
                ('base_frame', 'base_link'),
                ('map_frame', 'map'),
                ('use_detections', True),
                ('enable_visualization', True)
            ]
        )
        
        # 获取参数
        self.vehicle_width = self.get_parameter('vehicle_width').value
        self.vehicle_length = self.get_parameter('vehicle_length').value
        self.planning_rate = self.get_parameter('planning_rate').value
        self.max_planning_distance = self.get_parameter('max_planning_distance').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.use_detections = self.get_parameter('use_detections').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        
        # CV桥接器
        self.cv_bridge = CvBridge()
        
        # TF缓冲区和监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 数据锁
        self.data_lock = Lock()
        
        # 数据缓存
        self.latest_depth = None
        self.latest_rgb = None
        self.latest_camera_info = None
        self.latest_detections = None
        self.current_target = None
        self.current_vehicle_state = None
        self.motion_planner = None
        
        # 初始化订阅者
        self.setup_subscribers()
        
        # 初始化发布者
        self.setup_publishers()
        
        # 定时器
        self.planning_timer = self.create_timer(
            1.0 / self.planning_rate, 
            self.planning_callback
        )
        
        self.get_logger().info('Motion Planning Node initialized')
        
    def setup_subscribers(self):
        """设置订阅者"""
        # 深度图像订阅
        self.depth_sub = self.create_subscription(
            Image,
            '/depth/image_raw',
            self.depth_callback,
            10
        )
        
        # RGB图像订阅
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10
        )
        
        # 相机信息订阅
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # 检测结果订阅（可选）
        if self.use_detections:
            self.detection_sub = self.create_subscription(
                Detection2DArray,
                '/detections',
                self.detection_callback,
                10
            )
        
        # 目标点订阅
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        # 车辆位置订阅（从定位系统）
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        
    def setup_publishers(self):
        """设置发布者"""
        # 路径发布
        self.path_pub = self.create_publisher(
            Path,
            '/planned_path',
            10
        )
        
        # 控制命令发布
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 占用栅格发布
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/local_costmap',
            10
        )
        
        # 调试信息发布
        self.debug_image_pub = self.create_publisher(
            Image,
            '/motion_planning/debug_image',
            10
        )
        
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            with self.data_lock:
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    
    def rgb_callback(self, msg):
        """RGB图像回调"""
        try:
            with self.data_lock:
                self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')
    
    def camera_info_callback(self, msg):
        """相机信息回调"""
        with self.data_lock:
            self.latest_camera_info = msg
            
        # 初始化motion planner（只需要初始化一次）
        if self.motion_planner is None:
            camera_params = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.motion_planner = DepthBasedMotionPlanner(camera_params)
            self.get_logger().info('Motion planner initialized with camera parameters')
    
    def detection_callback(self, msg):
        """检测结果回调"""
        with self.data_lock:
            self.latest_detections = msg
    
    def goal_callback(self, msg):
        """目标点回调"""
        with self.data_lock:
            # 转换目标点到base_link坐标系
            try:
                target_in_base = self.transform_pose(msg, self.base_frame)
                self.current_target = np.array([
                    target_in_base.pose.position.x,
                    target_in_base.pose.position.y
                ])
                self.get_logger().info(f'New target received: {self.current_target}')
            except Exception as e:
                self.get_logger().error(f'Error transforming goal: {str(e)}')
    
    def pose_callback(self, msg):
        """车辆位置回调"""
        with self.data_lock:
            try:
                # 转换到base_link坐标系
                pose_in_base = self.transform_pose_with_covariance(msg, self.base_frame)
                
                # 提取位置和方向
                position = np.array([
                    pose_in_base.pose.pose.position.x,
                    pose_in_base.pose.pose.position.y,
                    pose_in_base.pose.pose.position.z
                ])
                
                # 从四元数计算航向角
                orientation = self.quaternion_to_yaw(pose_in_base.pose.pose.orientation)
                
                # 估算速度（简化处理）
                velocity = 0.0  # 可以通过历史位置计算
                
                self.current_vehicle_state = VehicleState(
                    position=position,
                    orientation=orientation,
                    velocity=velocity,
                    width=self.vehicle_width,
                    length=self.vehicle_length
                )
                
            except Exception as e:
                self.get_logger().error(f'Error processing pose: {str(e)}')
    
    def planning_callback(self):
        """路径规划主回调"""
        # 检查是否有必要的数据
        if not self.check_data_availability():
            return
        
        try:
            with self.data_lock:
                # 复制数据以避免竞争条件
                depth_map = self.latest_depth.copy()
                rgb_image = self.latest_rgb.copy() if self.latest_rgb is not None else None
                target = self.current_target.copy() if self.current_target is not None else None
                vehicle_state = self.current_vehicle_state
                detections = self.latest_detections
            
            if target is None or vehicle_state is None:
                return
            
            # 生成缺陷掩码
            defect_masks = self.generate_defect_masks(detections, depth_map.shape)
            
            # 执行路径规划
            result = self.motion_planner.process_frame(
                rgb_image if rgb_image is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                depth_map,
                defect_masks,
                vehicle_state,
                target
            )
            
            # 发布结果
            self.publish_results(result)
            
        except Exception as e:
            self.get_logger().error(f'Error in planning callback: {str(e)}')
    
    def check_data_availability(self):
        """检查数据可用性"""
        with self.data_lock:
            if self.latest_depth is None:
                return False
            if self.motion_planner is None:
                return False
            if self.current_vehicle_state is None:
                # 尝试从TF获取车辆状态
                self.update_vehicle_state_from_tf()
                if self.current_vehicle_state is None:
                    return False
        return True
    
    def update_vehicle_state_from_tf(self):
        """从TF更新车辆状态"""
        try:
            # 获取base_link到map的变换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            orientation = self.quaternion_to_yaw(transform.transform.rotation)
            
            self.current_vehicle_state = VehicleState(
                position=position,
                orientation=orientation,
                velocity=0.0,  # 简化处理
                width=self.vehicle_width,
                length=self.vehicle_length
            )
            
        except Exception as e:
            self.get_logger().debug(f'Could not get vehicle state from TF: {str(e)}')
    
    def generate_defect_masks(self, detections, image_shape):
        """从检测结果生成缺陷掩码"""
        masks = []
        
        if detections is None or not self.use_detections:
            # 如果没有检测结果，可以使用基于深度的简单检测
            return self.generate_depth_based_masks(image_shape)
        
        for detection in detections.detections:
            # 从检测框创建掩码
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            
            # 获取边界框
            bbox = detection.bbox
            x_min = int(bbox.center.x - bbox.size_x / 2)
            y_min = int(bbox.center.y - bbox.size_y / 2)
            x_max = int(bbox.center.x + bbox.size_x / 2)
            y_max = int(bbox.center.y + bbox.size_y / 2)
            
            # 确保边界框在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_shape[1], x_max)
            y_max = min(image_shape[0], y_max)
            
            # 创建掩码
            mask[y_min:y_max, x_min:x_max] = 255
            masks.append(mask)
        
        return masks
    
    def generate_depth_based_masks(self, image_shape):
        """基于深度信息生成简单的缺陷掩码"""
        masks = []
        
        with self.data_lock:
            if self.latest_depth is None:
                return masks
            
            depth_copy = self.latest_depth.copy()
        
        # 简单的深度变化检测
        # 计算深度梯度
        grad_x = cv2.Sobel(depth_copy, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_copy, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 阈值化
        threshold = np.percentile(gradient_magnitude[gradient_magnitude > 0], 95)
        defect_mask = (gradient_magnitude > threshold).astype(np.uint8) * 255
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找连通区域
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 过滤小区域
                mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                masks.append(mask)
        
        return masks
    
    def publish_results(self, planning_result):
        """发布规划结果"""
        try:
            # 发布路径
            self.publish_path(planning_result['planned_path'])
            
            # 发布控制命令
            self.publish_control_commands(planning_result['control_commands'])
            
            # 发布占用栅格
            self.publish_occupancy_grid(planning_result['occupancy_grid'])
            
            # 发布调试图像
            if self.enable_visualization:
                self.publish_debug_image(planning_result)
                
        except Exception as e:
            self.get_logger().error(f'Error publishing results: {str(e)}')
    
    def publish_path(self, path):
        """发布路径"""
        if not path:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = self.base_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(point[0])
            pose_stamped.pose.position.y = float(point[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)
    
    def publish_control_commands(self, commands):
        """发布控制命令"""
        if not commands:
            return
        
        cmd_vel = Twist()
        cmd_vel.linear.x = commands.get('throttle', 0.0) * 5.0  # 转换为速度
        cmd_vel.angular.z = commands.get('steering', 0.0)
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def publish_occupancy_grid(self, occupancy_grid):
        """发布占用栅格"""
        if occupancy_grid is None:
            return
        
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = self.base_frame
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 设置栅格信息
        grid_msg.info.resolution = 0.1  # 10cm分辨率
        grid_msg.info.width = occupancy_grid.shape[1]
        grid_msg.info.height = occupancy_grid.shape[0]
        grid_msg.info.origin.position.x = -grid_msg.info.width * grid_msg.info.resolution / 2
        grid_msg.info.origin.position.y = -grid_msg.info.height * grid_msg.info.resolution / 2
        grid_msg.info.origin.orientation.w = 1.0
        
        # 转换占用栅格数据
        grid_data = occupancy_grid.flatten()
        grid_data = (grid_data * 100).astype(np.int8)  # 转换为0-100的占用概率
        grid_msg.data = grid_data.tolist()
        
        self.occupancy_pub.publish(grid_msg)
    
    def publish_debug_image(self, planning_result):
        """发布调试图像"""
        try:
            with self.data_lock:
                if self.latest_rgb is None:
                    return
                debug_image = self.latest_rgb.copy()
            
            # 在图像上绘制检测结果和路径
            # 这里可以添加可视化代码
            
            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header.frame_id = self.camera_frame
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing debug image: {str(e)}')
    
    def transform_pose(self, pose_stamped, target_frame):
        """变换位姿到目标坐标系"""
        return self.tf_buffer.transform(pose_stamped, target_frame)
    
    def transform_pose_with_covariance(self, pose_with_cov, target_frame):
        """变换带协方差的位姿到目标坐标系"""
        pose_stamped = PoseStamped()
        pose_stamped.header = pose_with_cov.header
        pose_stamped.pose = pose_with_cov.pose.pose
        
        transformed = self.tf_buffer.transform(pose_stamped, target_frame)
        
        result = PoseWithCovarianceStamped()
        result.header = transformed.header
        result.pose.pose = transformed.pose
        result.pose.covariance = pose_with_cov.pose.covariance
        
        return result
    
    def quaternion_to_yaw(self, quaternion):
        """从四元数计算航向角"""
        import math
        
        # 提取四元数分量
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        
        # 计算航向角
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw

def main(args=None):
    rclpy.init(args=args)
    
    node = MotionPlanningNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()