import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import copy
import time
directory = "data/open_door/episode_1/front_pcd"
save_path = os.path.join(os.path.dirname(directory), "frame.npy")
file = "data/open_door/episode_2/front_pcd/0.npy"


def get_cluster_color(labels,colors):
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1 labels)
    for label in unique_labels:
        color = colors[labels == label][0]  # Get the color for the current label
        print(f"Label {label}: Color {color[:3]}")  # Print only RGB channels


def save_frame(frames):
    frames = np.array(frames)
    np.save(save_path, frames)

def visualize_frame():
    pcd, obb = obb_from_pcd(file)
    original_obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent)
    original_axes = create_axes(original_obb)
    obb,R = adjust_obb(obb)
    axes = create_axes(obb)
    o3d.visualization.draw_geometries([pcd,original_obb,*original_axes])
    # o3d.visualization.draw_geometries([ pcd,*axes])

def visualize_obb_as_video():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    
    frames = []
    for filename in sorted(os.listdir(directory)):
        
        file_path = os.path.join(directory, filename)
        
        # Get the point cloud and OBB for the current file
        pcd, obb = obb_from_pcd(file_path)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        # Update the point cloud geometry with new points and colors
        if obb:
            vis.add_geometry(obb)
            vis.update_geometry(obb)
        
        new_obb = copy.deepcopy(obb)
        new_obb,R = adjust_obb(new_obb)
        frames.append(R)
        # Refresh visualization
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.5)
    return frames
    
    # vis.destroy_window()

def adjust_obb(obb):
    # Extract the current rotation matrix from OBB
    R_original = obb.R

    # Step 1: Rotate 90 degrees around the y-axis to map x -> z, z -> -x
    rotation_matrix_y_90 = np.array([
        [0, 0, 1],  # x-axis maps to z-axis
        [0, 1, 0],  # y-axis remains y-axis
        [-1, 0, 0]  # z-axis maps to -x-axis
    ])

    # Step 2: Rotate 180 degrees around the new x-axis to map y -> -y, z -> -z
    rotation_matrix_x_180 = np.array([
        [-1, 0, 0],  # x-axis remains x-axis
        [0, -1, 0], # y-axis flips to -y
        [0, 0, 1]  # z-axis flips to -z
    ])

    # Apply the rotations sequentially to obtain the adjusted rotation matrix
    # R_adjusted = R_original @ rotation_matrix_y_90 @ rotation_matrix_x_180
    R_adjusted = R_original@ rotation_matrix_y_90 @ rotation_matrix_x_180
    # Update the OBB rotation matrix with the adjusted rotation
    obb.R = R_adjusted

    return obb, obb.R

def create_axes(obb):
    # Extract the OBB center and rotation matrix
    center = obb.center
    R = obb.R
    extents = obb.extent / 2  # Half-lengths along each axis

    # Define the axes based on OBB's rotation matrix and extent
    x_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 0] * extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    x_axis.colors = o3d.utility.Vector3dVector([(1, 0, 0)])  # Red for x-axis

    y_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 1] * extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    y_axis.colors = o3d.utility.Vector3dVector([(0, 1, 0)])  # Green for y-axis

    z_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([center, center + R[:, 2] * 2*extents[0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    z_axis.colors = o3d.utility.Vector3dVector([(0, 0, 1)])  # Blue for z-axis

    return [x_axis, y_axis, z_axis]


def obb_from_pcd(file_path):
    
    points = np.load(file_path)
    points = points.reshape(-1, 3)
    print(f"shape:{points.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    #DBSCAN clustering and visualize
    labels = np.array(pcd.cluster_dbscan(eps=0.018, min_points=20, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0 
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

  
    target_color = np.array([0.12156863, 0.46666667, 0.70588235])  # Light blue in "tab20"

    # Find the label with the color closest to this blue
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1 labels)
    door_label = None
    for label in unique_labels:
        label_color = colors[labels == label][0][:3]  # Get the color for this label
        if np.allclose(label_color, target_color, atol=0.1):  # Allow a small tolerance
            door_label = label
            break

    if door_label is not None:
        # Extract points of the door based on the door_label
        door_points = points[labels == door_label]
        door_pcd = o3d.geometry.PointCloud()
        door_pcd.points = o3d.utility.Vector3dVector(door_points)

        # Calculate the OBB for the refined door points
        obb = door_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Set the OBB color to red
        return pcd,obb
    else:
        print("No label matched the door's color.")
        return pcd, None

   
def get_obb_parameter(obb):

 
    obb_center = obb.center  # Center of the OBB
    obb_extents = obb.extent  # Dimensions (lengths along the principal axes)
    obb_rotation_matrix = obb.R  # Rotation matrix (orientation of the OBB)

    # Print the OBB parameters
    print(f"OBB Center: {obb_center}")
    print(f"OBB Extents (Lengths along axes): {obb_extents}")
    print(f"OBB Rotation Matrix:\n {obb_rotation_matrix}")

def visualize_pcd():
    points = np.load("data/open_door/episode_1/front_pcd/0.npy")
    points = points.reshape(-1, 3)
    print(f"shape:{points.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def get_intrinsics():
    # Assuming 'env' is an instance of the class containing get_scene_data
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,JointPosition,EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    env = Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
    
    scene_data = env.get_scene_data()

    # Accessing data for specific cameras
    front_camera_data = scene_data.get("front_camera")
    
    # For example, to access the intrinsics of the front camera
    if front_camera_data is not None:
        front_intrinsics = front_camera_data["intrinsics"]
        front_near_plane = front_camera_data["near_plane"]
        front_far_plane = front_camera_data["far_plane"]
        front_extrinsics = front_camera_data["extrinsics"]

        print("Front Camera Intrinsics:\n", front_intrinsics)
        print("Front Camera Near Plane:", front_near_plane)
        print("Front Camera Far Plane:", front_far_plane)
        print("Front Camera Extrinsics:\n", front_extrinsics)


if __name__ == "__main__":
    visualize_frame()


