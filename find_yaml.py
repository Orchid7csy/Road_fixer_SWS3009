import os
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR, USER_CONFIG_DIR

def check_yolo_dataset_paths():
    """检查YOLO搜索数据集的所有路径"""
    print("YOLO 数据集搜索路径:")
    print("-" * 50)
    
    # 1. 当前工作目录
    current_dir = os.getcwd()
    yaml_in_current = os.path.join(current_dir, 'crack-seg.yaml')
    print(f"1. 当前目录: {current_dir}")
    print(f"   crack-seg.yaml 存在: {os.path.exists(yaml_in_current)}")
    
    # 2. YOLO 数据集目录
    print(f"\n2. YOLO 数据集目录: {DATASETS_DIR}")
    yaml_in_datasets = os.path.join(DATASETS_DIR, 'crack-seg.yaml')
    print(f"   crack-seg.yaml 存在: {os.path.exists(yaml_in_datasets)}")
    
    # 3. 用户配置目录
    print(f"\n3. 用户配置目录: {USER_CONFIG_DIR}")
    yaml_in_config = os.path.join(USER_CONFIG_DIR, 'crack-seg.yaml')
    print(f"   crack-seg.yaml 存在: {os.path.exists(yaml_in_config)}")
    
    # 4. 检查是否是预定义数据集
    print(f"\n4. 检查预定义数据集")
    try:
        # 尝试加载模型并获取数据集信息
        model = YOLO("yolo11n-seg.pt")
        
        # 这里不实际训练，只是检查数据集配置
        print("   尝试解析数据集配置...")
        
        # 检查 ultralytics 包中的预定义数据集
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import yaml_load
        
        # 尝试查找内置的数据集配置
        builtin_datasets = []
        try:
            # 搜索 ultralytics 包中的 yaml 文件
            import ultralytics
            ultralytics_dir = os.path.dirname(ultralytics.__file__)
            for root, dirs, files in os.walk(ultralytics_dir):
                for file in files:
                    if file.endswith('.yaml') and 'crack' in file.lower():
                        builtin_datasets.append(os.path.join(root, file))
        except Exception as e:
            print(f"   搜索内置数据集时出错: {e}")
        
        if builtin_datasets:
            print(f"   找到相关的内置数据集配置: {builtin_datasets}")
        else:
            print("   未找到相关的内置数据集配置")
            
    except Exception as e:
        print(f"   检查时出错: {e}")

def simulate_training_start():
    """模拟训练开始，查看YOLO实际使用的配置"""
    print("\n" + "=" * 60)
    print("模拟训练开始，查看YOLO行为")
    print("=" * 60)
    
    try:
        model = YOLO("yolo11n-seg.pt")
        
        # 尝试获取数据集信息（不实际训练）
        print("尝试解析 crack-seg.yaml...")
        
        # 这里可能会触发自动下载
        print("注意: 如果看到下载进度条，说明YOLO正在自动下载数据集")
        
        # 检查训练参数
        from ultralytics.cfg import get_cfg
        cfg = get_cfg()
        print(f"默认配置已加载")
        
    except Exception as e:
        print(f"模拟训练时出错: {e}")

def check_download_cache():
    """检查YOLO的下载缓存"""
    print("\n" + "=" * 60)
    print("检查YOLO下载缓存")
    print("=" * 60)
    
    # 检查常见的缓存位置
    cache_locations = [
        os.path.expanduser("~/.ultralytics/"),
        os.path.expanduser("~/.cache/ultralytics/"),
        USER_CONFIG_DIR,
        DATASETS_DIR
    ]
    
    for location in cache_locations:
        if os.path.exists(location):
            print(f"缓存目录存在: {location}")
            # 查找相关文件
            for root, dirs, files in os.walk(location):
                for file in files:
                    if 'crack' in file.lower() and file.endswith('.yaml'):
                        print(f"  找到文件: {os.path.join(root, file)}")
        else:
            print(f"缓存目录不存在: {location}")

if __name__ == "__main__":
    check_yolo_dataset_paths()
    simulate_training_start()
    check_download_cache()
    
    print("\n" + "=" * 60)
    print("总结:")
    print("- 如果训练能正常开始，说明YOLO找到了数据集配置")
    print("- 可能是自动下载的，也可能是预定义的数据集")
    print("- 检查训练输出日志，通常会显示使用的数据集路径")
    print("=" * 60)