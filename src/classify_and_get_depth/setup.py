from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'classify_and_get_depth'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 添加launch文件
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy]'))),
        # 添加配置文件
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'geometry_msgs',
        'cv_bridge',
        'numpy',
        'opencv-python',
        'ultralytics',
    ],
    zip_safe=True,  
    maintainer='csy',
    maintainer_email='2698452069@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crack_detector_node = classify_and_get_depth.nodes.crack_detector_node:main',
            'motion_planning_node = classify_and_get_depth.nodes.motion_planning_node:main'
        ],
    },    
)
