#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    package_dir = get_package_share_directory('road_fixer')
    config_dir = os.path.join(package_dir, 'config')
    
    # 声明启动参数
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(config_dir, 'crack-seg.yaml'),
        description='Path to the config file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # 深度发布节点
    depth_publisher_node = Node(
        package='road_fixer',
        executable='depth_publisher',
        name='depth_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            LaunchConfiguration('config_file')
        ]
    )
    
    # 裂缝检测节点
    crack_detector_node = Node(
        package='road_fixer',
        executable='crack_detector',
        name='crack_detector',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            LaunchConfiguration('config_file')
        ]
    )
    
    # 运动规划节点
    motion_planner_node = Node(
        package='road_fixer',
        executable='motion_planner',
        name='motion_planner',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            LaunchConfiguration('config_file')
        ]
    )
    
    # RViz可视化（可选）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=lambda context: context.launch_configurations.get('use_rviz', 'false') == 'true'
    )
    
    return LaunchDescription([
        config_file_arg,
        use_sim_time_arg,
        depth_publisher_node,
        crack_detector_node,
        motion_planner_node,
        # rviz_node,  # 如果需要可视化就取消注释
    ])