"""
Create simple OBJ geometry files for SPH simulations
"""

import numpy as np
from pathlib import Path

def create_box_obj(output_path, size=(0.2, 0.2, 0.2)):
    """
    Create a simple box OBJ file

    Args:
        output_path: Path to save .obj file
        size: (width, height, depth)
    """

    w, h, d = size
    hw, hh, hd = w/2, h/2, d/2

    # Vertices (8 corners of box)
    vertices = [
        [-hw, -hh, -hd],  # 0
        [ hw, -hh, -hd],  # 1
        [ hw,  hh, -hd],  # 2
        [-hw,  hh, -hd],  # 3
        [-hw, -hh,  hd],  # 4
        [ hw, -hh,  hd],  # 5
        [ hw,  hh,  hd],  # 6
        [-hw,  hh,  hd],  # 7
    ]

    # Faces (12 triangles, 2 per side)
    # Format: f v1 v2 v3 (1-indexed)
    faces = [
        # Front face
        [1, 2, 3], [1, 3, 4],
        # Back face
        [5, 7, 6], [5, 8, 7],
        # Left face
        [1, 4, 8], [1, 8, 5],
        # Right face
        [2, 6, 7], [2, 7, 3],
        # Top face
        [4, 3, 7], [3, 6, 7],
        # Bottom face
        [1, 5, 6], [1, 6, 2],
    ]

    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write("# Simple Box\n")
        f.write(f"# Size: {size}\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created box: {output_path}")

def create_sphere_obj(output_path, radius=0.1, resolution=16):
    """
    Create a simple sphere OBJ file using UV sphere

    Args:
        output_path: Path to save .obj file
        radius: Sphere radius
        resolution: Number of subdivisions
    """

    vertices = []
    faces = []

    # Generate vertices
    for i in range(resolution + 1):
        theta = np.pi * i / resolution  # 0 to pi

        for j in range(resolution):
            phi = 2 * np.pi * j / resolution  # 0 to 2pi

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            vertices.append([x, y, z])

    # Generate faces
    for i in range(resolution):
        for j in range(resolution):
            # Current quad vertices (1-indexed for OBJ)
            v1 = i * resolution + j + 1
            v2 = i * resolution + (j + 1) % resolution + 1
            v3 = (i + 1) * resolution + (j + 1) % resolution + 1
            v4 = (i + 1) * resolution + j + 1

            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write("# Simple Sphere\n")
        f.write(f"# Radius: {radius}, Resolution: {resolution}\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created sphere: {output_path}")

def create_cylinder_obj(output_path, radius=0.05, height=0.3, resolution=16):
    """
    Create a simple cylinder OBJ file

    Args:
        output_path: Path to save .obj file
        radius: Cylinder radius
        height: Cylinder height
        resolution: Number of circumferential subdivisions
    """

    vertices = []
    faces = []

    # Bottom cap center
    vertices.append([0, -height/2, 0])
    bottom_center_idx = 1

    # Bottom cap vertices
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([x, -height/2, z])

    # Top cap vertices
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([x, height/2, z])

    # Top cap center
    vertices.append([0, height/2, 0])
    top_center_idx = len(vertices)

    # Bottom cap faces
    for i in range(resolution):
        v1 = bottom_center_idx
        v2 = i + 2
        v3 = (i + 1) % resolution + 2
        faces.append([v1, v2, v3])

    # Side faces
    for i in range(resolution):
        bottom1 = i + 2
        bottom2 = (i + 1) % resolution + 2
        top1 = i + resolution + 2
        top2 = (i + 1) % resolution + resolution + 2

        # Two triangles per side quad
        faces.append([bottom1, bottom2, top2])
        faces.append([bottom1, top2, top1])

    # Top cap faces
    for i in range(resolution):
        v1 = top_center_idx
        v2 = i + resolution + 2
        v3 = (i + 1) % resolution + resolution + 2
        faces.append([v1, v3, v2])  # Reversed for correct normal

    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write("# Simple Cylinder\n")
        f.write(f"# Radius: {radius}, Height: {height}, Resolution: {resolution}\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Created cylinder: {output_path}")

if __name__ == "__main__":
    # Create geometry directory
    geom_dir = Path("datasets/geometries")
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Create various geometries
    create_box_obj(geom_dir / "box_small.obj", size=(0.15, 0.15, 0.15))
    create_box_obj(geom_dir / "box_medium.obj", size=(0.2, 0.2, 0.2))
    create_box_obj(geom_dir / "box_large.obj", size=(0.3, 0.3, 0.3))

    create_sphere_obj(geom_dir / "sphere_small.obj", radius=0.08, resolution=16)
    create_sphere_obj(geom_dir / "sphere_medium.obj", radius=0.1, resolution=20)
    create_sphere_obj(geom_dir / "sphere_large.obj", radius=0.12, resolution=24)

    create_cylinder_obj(geom_dir / "cylinder_thin.obj", radius=0.04, height=0.25, resolution=16)
    create_cylinder_obj(geom_dir / "cylinder_medium.obj", radius=0.06, height=0.3, resolution=20)
    create_cylinder_obj(geom_dir / "cylinder_thick.obj", radius=0.08, height=0.35, resolution=24)

    print()
    print("=" * 70)
    print(f"All geometries created in: {geom_dir}")
    print("=" * 70)
