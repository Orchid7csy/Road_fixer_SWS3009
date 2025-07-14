import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class DefectSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class VehicleState:
    position: np.ndarray  # [x, y, z]
    orientation: float    # yaw angle in radians
    velocity: float       # m/s
    width: float         # vehicle width
    length: float        # vehicle length

@dataclass
class DefectInfo:
    center_3d: np.ndarray     # 3D center position
    max_depth: float          # maximum depth
    area: float               # defect area in m²
    severity: DefectSeverity  # severity level
    bounding_box: np.ndarray  # 3D bounding box
    points_3d: np.ndarray     # 3D point cloud of defect

class DepthBasedMotionPlanner:
    def __init__(self, camera_params: Dict):
        """
        Initialize the motion planner
        
        Args:
            camera_params: Dictionary containing camera intrinsics
                          {'fx': focal_x, 'fy': focal_y, 'cx': center_x, 'cy': center_y}
        """
        self.camera_params = camera_params
        self.vehicle_safe_distance = 0.5  # meters
        self.max_depth_threshold = 0.1    # 10cm depth threshold
        self.critical_depth_threshold = 0.15  # 15cm critical threshold
        self.grid_resolution = 0.1  # 10cm grid resolution
        self.planning_horizon = 20.0  # 20m planning horizon
        
    def process_frame(self, 
                     rgb_image: np.ndarray, 
                     depth_map: np.ndarray,
                     defect_masks: List[np.ndarray],
                     vehicle_state: VehicleState,
                     target_waypoint: np.ndarray) -> Dict:
        """
        Complete processing pipeline for one frame
        
        Args:
            rgb_image: RGB image from camera
            depth_map: Depth map from DepthPro or similar network
            defect_masks: List of binary masks for detected defects
            vehicle_state: Current vehicle state
            target_waypoint: Target position [x, y]
            
        Returns:
            Dictionary containing defects info and planned path
        """
        # Step 1: Extract 3D defect information
        defects = self.extract_defects_3d(depth_map, defect_masks)
        
        # Step 2: Build occupancy grid
        occupancy_grid = self.build_occupancy_grid(defects, vehicle_state)
        
        # Step 3: Plan path
        planned_path = self.plan_path(vehicle_state, target_waypoint, occupancy_grid)
        
        # Step 4: Generate control commands
        control_commands = self.generate_control_commands(planned_path, vehicle_state)
        
        return {
            'defects': defects,
            'occupancy_grid': occupancy_grid,
            'planned_path': planned_path,
            'control_commands': control_commands
        }
    
    def extract_defects_3d(self, depth_map: np.ndarray, defect_masks: List[np.ndarray]) -> List[DefectInfo]:
        """
        Extract 3D information from depth map and defect masks
        """
        defects = []
        
        for mask in defect_masks:
            # Get depth values within the mask
            mask_coords = np.where(mask > 0)
            if len(mask_coords[0]) == 0:
                continue
                
            # Convert 2D pixels to 3D points
            points_3d = self.pixels_to_3d(mask_coords, depth_map)
            
            if len(points_3d) == 0:
                continue
            
            # Calculate defect properties
            defect_info = self.analyze_defect_3d(points_3d, mask)
            defects.append(defect_info)
            
        return defects
    
    def pixels_to_3d(self, mask_coords: Tuple, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert 2D pixel coordinates to 3D world coordinates
        """
        v_coords, u_coords = mask_coords
        depths = depth_map[v_coords, u_coords]
        
        # Filter out invalid depths
        valid_mask = (depths > 0) & (depths < 100)  # reasonable depth range
        if not np.any(valid_mask):
            return np.array([])
        
        u_coords = u_coords[valid_mask]
        v_coords = v_coords[valid_mask]
        depths = depths[valid_mask]
        
        # Convert to 3D coordinates (camera coordinate system)
        fx, fy = self.camera_params['fx'], self.camera_params['fy']
        cx, cy = self.camera_params['cx'], self.camera_params['cy']
        
        x = (u_coords - cx) * depths / fx
        y = (v_coords - cy) * depths / fy
        z = depths
        
        points_3d = np.column_stack([x, y, z])
        return points_3d
    
    def analyze_defect_3d(self, points_3d: np.ndarray, mask: np.ndarray) -> DefectInfo:
        """
        Analyze 3D defect properties
        """
        # Calculate center
        center_3d = np.mean(points_3d, axis=0)
        
        # Calculate max depth (assuming road surface is at z=0)
        max_depth = np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])
        
        # Calculate area (rough approximation)
        area = len(points_3d) * (self.grid_resolution ** 2)
        
        # Determine severity
        severity = self.assess_severity(max_depth, area)
        
        # Calculate 3D bounding box
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        bounding_box = np.array([min_coords, max_coords])
        
        return DefectInfo(
            center_3d=center_3d,
            max_depth=max_depth,
            area=area,
            severity=severity,
            bounding_box=bounding_box,
            points_3d=points_3d
        )
    
    def assess_severity(self, max_depth: float, area: float) -> DefectSeverity:
        """
        Assess defect severity based on depth and area
        """
        if max_depth > self.critical_depth_threshold:
            return DefectSeverity.CRITICAL
        elif max_depth > self.max_depth_threshold:
            if area > 0.5:  # Large area
                return DefectSeverity.HIGH
            else:
                return DefectSeverity.MEDIUM
        else:
            return DefectSeverity.LOW
    
    def build_occupancy_grid(self, defects: List[DefectInfo], vehicle_state: VehicleState) -> np.ndarray:
        """
        Build occupancy grid for path planning
        """
        # Define grid boundaries
        grid_width = int(self.planning_horizon / self.grid_resolution)
        grid_height = int(self.planning_horizon / self.grid_resolution)
        
        # Initialize grid (0 = free, 1 = occupied)
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Vehicle position as origin
        origin_x, origin_y = vehicle_state.position[0], vehicle_state.position[1]
        
        for defect in defects:
            # Skip low severity defects
            if defect.severity == DefectSeverity.LOW:
                continue
                
            # Convert defect position to grid coordinates
            defect_x, defect_y = defect.center_3d[0], defect.center_3d[1]
            
            # Calculate grid position relative to vehicle
            grid_x = int((defect_x - origin_x) / self.grid_resolution + grid_width // 2)
            grid_y = int((defect_y - origin_y) / self.grid_resolution + grid_height // 2)
            
            # Add safety margin based on severity
            if defect.severity == DefectSeverity.CRITICAL:
                margin = int(self.vehicle_safe_distance * 2 / self.grid_resolution)
            else:
                margin = int(self.vehicle_safe_distance / self.grid_resolution)
            
            # Mark occupied cells
            for dy in range(-margin, margin + 1):
                for dx in range(-margin, margin + 1):
                    gx, gy = grid_x + dx, grid_y + dy
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        occupancy_grid[gy, gx] = 1
        
        return occupancy_grid
    
    def plan_path(self, 
                 vehicle_state: VehicleState, 
                 target_waypoint: np.ndarray,
                 occupancy_grid: np.ndarray) -> List[np.ndarray]:
        """
        Plan path using A* algorithm
        """
        grid_height, grid_width = occupancy_grid.shape
        
        # Convert positions to grid coordinates
        origin_x, origin_y = vehicle_state.position[0], vehicle_state.position[1]
        
        start_grid = (grid_height // 2, grid_width // 2)  # Vehicle at center
        
        target_grid_x = int((target_waypoint[0] - origin_x) / self.grid_resolution + grid_width // 2)
        target_grid_y = int((target_waypoint[1] - origin_y) / self.grid_resolution + grid_height // 2)
        target_grid = (target_grid_y, target_grid_x)
        
        # A* pathfinding
        path_grid = self.a_star_search(occupancy_grid, start_grid, target_grid)
        
        if not path_grid:
            # No path found, return straight line (emergency)
            return [vehicle_state.position[:2], target_waypoint]
        
        # Convert grid path back to world coordinates
        path_world = []
        for gy, gx in path_grid:
            world_x = origin_x + (gx - grid_width // 2) * self.grid_resolution
            world_y = origin_y + (gy - grid_height // 2) * self.grid_resolution
            path_world.append(np.array([world_x, world_y]))
        
        # Smooth the path
        smoothed_path = self.smooth_path(path_world)
        
        return smoothed_path
    
    def a_star_search(self, grid: np.ndarray, start: Tuple, goal: Tuple) -> List[Tuple]:
        """
        A* pathfinding algorithm
        """
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = pos[0] + dy, pos[1] + dx
                    if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and 
                        grid[ny, nx] == 0):
                        neighbors.append((ny, nx))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Smooth the path using spline interpolation
        """
        if len(path) < 3:
            return path
        
        # Extract x and y coordinates
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # Spline interpolation
        try:
            tck, u = splprep([x, y], s=0, k=min(3, len(path)-1))
            u_new = np.linspace(0, 1, len(path) * 2)  # Increase resolution
            x_new, y_new = splev(u_new, tck)
            
            smoothed_path = [np.array([x_new[i], y_new[i]]) for i in range(len(x_new))]
            return smoothed_path
        except:
            return path  # Return original if smoothing fails
    
    def generate_control_commands(self, 
                                path: List[np.ndarray], 
                                vehicle_state: VehicleState) -> Dict:
        """
        Generate control commands for the vehicle
        """
        if len(path) < 2:
            return {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}
        
        # Calculate steering angle
        current_pos = vehicle_state.position[:2]
        target_pos = path[1]  # Next waypoint
        
        # Vector from current position to target
        direction_vector = target_pos - current_pos
        target_angle = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Steering error
        angle_error = target_angle - vehicle_state.orientation
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize
        
        # Simple P controller for steering
        steering_gain = 2.0
        steering_command = np.clip(steering_gain * angle_error, -1.0, 1.0)
        
        # Speed control based on path curvature
        if len(path) > 2:
            curvature = self.calculate_curvature(path[:3])
            max_speed = 10.0  # m/s
            speed_factor = max(0.3, 1.0 - curvature)  # Slow down on curves
            target_speed = max_speed * speed_factor
        else:
            target_speed = 5.0
        
        # Simple speed control
        speed_error = target_speed - vehicle_state.velocity
        if speed_error > 0:
            throttle = min(0.5, speed_error * 0.1)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.5, -speed_error * 0.1)
        
        return {
            'steering': steering_command,
            'throttle': throttle,
            'brake': brake,
            'target_speed': target_speed
        }
    
    def calculate_curvature(self, path_segment: List[np.ndarray]) -> float:
        """
        Calculate path curvature for speed control
        """
        if len(path_segment) < 3:
            return 0.0
        
        p1, p2, p3 = path_segment[:3]
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0.0
        
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Return curvature (higher values = more curved)
        return angle / np.pi
    
    def visualize_planning(self, 
                          occupancy_grid: np.ndarray,
                          path: List[np.ndarray],
                          defects: List[DefectInfo],
                          vehicle_state: VehicleState):
        """
        Visualize the planning result
        """
        plt.figure(figsize=(12, 8))
        
        # Show occupancy grid
        plt.subplot(2, 2, 1)
        plt.imshow(occupancy_grid, cmap='gray', origin='lower')
        plt.title('Occupancy Grid')
        plt.colorbar()
        
        # Show path in world coordinates
        plt.subplot(2, 2, 2)
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Planned Path')
        
        # Plot defects
        for defect in defects:
            color = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
            plt.scatter(defect.center_3d[0], defect.center_3d[1], 
                       c=color[defect.severity.value], s=100, alpha=0.7)
        
        # Plot vehicle
        plt.scatter(vehicle_state.position[0], vehicle_state.position[1], 
                   c='blue', s=200, marker='s', label='Vehicle')
        
        plt.title('Path Planning Result')
        plt.legend()
        plt.axis('equal')
        
        # Show defect statistics
        plt.subplot(2, 2, 3)
        severities = [defect.severity.value for defect in defects]
        severity_counts = {s: severities.count(s) for s in ['low', 'medium', 'high', 'critical']}
        plt.bar(severity_counts.keys(), severity_counts.values())
        plt.title('Defect Severity Distribution')
        
        # Show depth distribution
        plt.subplot(2, 2, 4)
        depths = [defect.max_depth for defect in defects]
        if depths:
            plt.hist(depths, bins=10, alpha=0.7)
            plt.axvline(self.max_depth_threshold, color='orange', linestyle='--', label='Warning')
            plt.axvline(self.critical_depth_threshold, color='red', linestyle='--', label='Critical')
            plt.xlabel('Max Depth (m)')
            plt.ylabel('Count')
            plt.title('Defect Depth Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Camera parameters (example values)
    camera_params = {
        'fx': 800.0,
        'fy': 800.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # Initialize motion planner
    planner = DepthBasedMotionPlanner(camera_params)
    
    # Example vehicle state
    vehicle_state = VehicleState(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=0.0,
        velocity=5.0,
        width=1.8,
        length=4.5
    )
    
    # Example target waypoint
    target_waypoint = np.array([10.0, 2.0])
    
    print("Motion planner initialized successfully!")
    print("Ready to process frames with depth maps and defect masks.")