from setuptools import find_packages, setup

package_name = 'depth_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'depth_publisher_node = classify_and_get_depth.nodes.depth_publisher_node:main',            
        ],
    },
)
