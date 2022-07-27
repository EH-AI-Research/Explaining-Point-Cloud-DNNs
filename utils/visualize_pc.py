import open3d as o3d

pcd = o3d.io.read_point_cloud("../pc-images/qualitative-point-dropping-and-merging/plane/pc-pn1-0-1.ply", format='auto', print_progress=False)
o3d.visualization.draw_geometries([pcd])