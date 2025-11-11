import open3d as o3d
import numpy as np
import copy

def print_info(geometry, step_name):
    """Print information about geometry"""
    print(f"\n=== {step_name} ===")

    if isinstance(geometry, o3d.geometry.TriangleMesh):
        print(f"Number of vertices: {len(geometry.vertices)}")
        print(f"Number of triangles: {len(geometry.triangles)}")
        print(f"Has vertex colors: {geometry.has_vertex_colors()}")
        print(f"Has vertex normals: {geometry.has_vertex_normals()}")
    elif isinstance(geometry, o3d.geometry.PointCloud):
        print(f"Number of points: {len(geometry.points)}")
        print(f"Has colors: {geometry.has_colors()}")
        print(f"Has normals: {geometry.has_normals()}")
    elif isinstance(geometry, o3d.geometry.VoxelGrid):
        try:
            voxels = geometry.get_voxels()
            print(f"Number of voxels: {len(voxels)}")
        except Exception as e:
            print(f"Number of voxels: Could not access count ({e})")
    else:
        print(f"Geometry type {type(geometry).__name__} is not fully handled in print_info.")

def main():
    # Task 1: Loading and Visualization
    print("\n" + "="*50)
    print("TASK 1: LOADING AND VISUALIZATION")
    print("="*50)
    
    mesh_path = r"C:\Users\а\Downloads\Skull_v3_L2.123c1407fc1e-ea5c-4cb9-9072-d28b8aba4c36\Skull_v3_L2.123c1407fc1e-ea5c-4cb9-9072-d28b8aba4c36\12140_Skull_v3_L2.obj"
    
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) == 0:
            raise IOError("Mesh is empty, trying fallback.")
    except (UnicodeDecodeError, IOError):
        print(f"Warning: Failed to load '{mesh_path}' directly due to encoding or read issue. Trying fallback method...")
        with open(mesh_path, 'r', encoding='latin-1') as f:
            obj_content = f.read()
        
        with open("temp_mesh.obj", "w", encoding="utf-8") as f:
            f.write(obj_content)
        mesh = o3d.io.read_triangle_mesh("temp_mesh.obj")
    
    if len(mesh.vertices) == 0:
        print(f"Failed to load mesh from '{mesh_path}', creating a sample cube instead...")
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.translate([-0.5, -0.5, -0.5])  # Center it
    
    print_info(mesh, "Original Model")
    
    print("Displaying original model...")
    o3d.visualization.draw_geometries([mesh], window_name="Task 1: Original Model")
    
    # Task 2: Conversion to Point Cloud
    print("\n" + "="*50)
    print("TASK 2: CONVERSION TO POINT CLOUD")
    print("="*50)
    
    print("Sampling points from mesh...")
    point_cloud = mesh.sample_points_uniformly(number_of_points=15000)
    
    print_info(point_cloud, "Point Cloud")
    
    print("Displaying point cloud...")
    o3d.visualization.draw_geometries([point_cloud], window_name="Task 2: Point Cloud")
    
    # Task 3: Surface Reconstruction from Point Cloud
    print("\n" + "="*50)
    print("TASK 3: SURFACE RECONSTRUCTION FROM POINT CLOUD")
    print("="*50)
    
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    point_cloud.orient_normals_consistent_tangent_plane(k=30)
    
    print("Performing Poisson surface reconstruction...")
    mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=6, width=0, scale=1.1, linear_fit=False)
    
    # Crop artifacts using the bounding box of the original point cloud, as per the assignment.
    bbox = point_cloud.get_axis_aligned_bounding_box()
    mesh_reconstructed = mesh_reconstructed.crop(bbox)
    
    mesh_reconstructed.remove_degenerate_triangles()
    mesh_reconstructed.remove_duplicated_triangles()
    mesh_reconstructed.remove_duplicated_vertices()
    mesh_reconstructed.remove_non_manifold_edges()
    
    print_info(mesh_reconstructed, "Reconstructed Mesh")
    
    print("Displaying reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh_reconstructed], window_name="Task 3: Reconstructed Mesh")
    
    # Task 4: Voxelization
    print("\n" + "="*50)
    print("TASK 4: VOXELIZATION")
    print("="*50)
    
    voxel_size = 0.05
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
    
    print_info(voxel_grid, f"Voxel Grid (size={voxel_size})")
    
    print("Displaying voxel grid...")
    o3d.visualization.draw_geometries([voxel_grid], window_name="Task 4: Voxel Grid")
    
    # Task 5: Adding a Plane
    print("\n" + "="*50)
    print("TASK 5: ADDING A PLANE")
    print("="*50)
    
    points = np.asarray(point_cloud.points)
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    padding = 0.2
    height = (y_max - y_min) + 2 * padding
    depth = (z_max - z_min) + 2 * padding
    
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=0.02, height=height, depth=depth)
    
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    plane_mesh.translate([-0.01, center_y - height/2, center_z - depth/2])
    
    plane_mesh.paint_uniform_color([0.6, 0.6, 0.6])  # Lighter gray for visibility
    
    print("Created vertical clipping plane at x=0")
    print(f"Plane vertices: {len(plane_mesh.vertices)}")
    print(f"Plane triangles: {len(plane_mesh.triangles)}")
    print(f"Plane dimensions: {0.02:.3f} x {height:.3f} x {depth:.3f}")
    
    print("Displaying object with clipping plane...")
    o3d.visualization.draw_geometries([point_cloud, plane_mesh], window_name="Task 5: Object with Clipping Plane")
    
    # Task 6: Surface Clipping
    print("\n" + "="*50)
    print("TASK 6: SURFACE CLIPPING")
    print("="*50)
    
    points = np.asarray(point_cloud.points)
    mask = points[:, 0] <= 0
    clipped_cloud = o3d.geometry.PointCloud()
    clipped_cloud.points = o3d.utility.Vector3dVector(points[mask])
    if point_cloud.has_normals():
        clipped_cloud.normals = o3d.utility.Vector3dVector(np.asarray(point_cloud.normals)[mask])
    
    print_info(clipped_cloud, "Clipped Point Cloud")
    print(f"Original points: {len(point_cloud.points)}")
    print(f"Remaining points after clipping: {len(clipped_cloud.points)}")
    
    print("Displaying clipped model...")
    o3d.visualization.draw_geometries([clipped_cloud], window_name="Task 6: Clipped Model")
    
    # Task 7: Working with Color and Extremes
    print("\n" + "="*50)
    print("TASK 7: WORKING WITH COLOR AND EXTREMES")
    print("="*50)
    
    points = np.asarray(point_cloud.points)
    
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    
    normalized_z = (z_values - z_min) / (z_max - z_min) if z_max != z_min else np.zeros_like(z_values)
    
    colors = np.zeros((len(points), 3))
    colors[:, 0] = normalized_z        # Red increases with Z
    colors[:, 2] = 1.0 - normalized_z  # Blue decreases with Z
    
    colored_cloud = copy.deepcopy(point_cloud)
    colored_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    z_min_idx = np.argmin(z_values)
    z_max_idx = np.argmax(z_values)
    
    min_point = points[z_min_idx]
    max_point = points[z_max_idx]
    
    print(f"Z-axis extremes:")
    print(f"Minimum Z point: {min_point} (Z = {z_min:.3f})")
    print(f"Maximum Z point: {max_point} (Z = {z_max:.3f})")
    
    min_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    min_sphere.translate(min_point)
    min_sphere.paint_uniform_color([0, 1, 0])  # Green for minimum
    
    max_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    max_sphere.translate(max_point)
    max_sphere.paint_uniform_color([1, 0, 1])  # Magenta for maximum
    
    print("Displaying colored point cloud with extreme points marked...")
    o3d.visualization.draw_geometries([colored_cloud, min_sphere, max_sphere], 
                                    window_name="Task 7: Colored Cloud with Extremes")
    
    print("\n" + "="*50)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*50)
    
if __name__ == "__main__":
    main()