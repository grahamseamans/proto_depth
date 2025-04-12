"""Analyze dimensions of an OBJ model file"""


def analyze_obj(filepath):
    with open(filepath) as f:
        vertices = [line.split()[1:] for line in f if line.startswith("v ")]
        verts = [[float(x) for x in v] for v in vertices]
        min_coords = [min(v[i] for v in verts) for i in range(3)]
        max_coords = [max(v[i] for v in verts) for i in range(3)]
        dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
        center = [(max_coords[i] + min_coords[i]) / 2 for i in range(3)]

        print(f"\nModel analysis for {filepath}:")
        print(f"Dimensions (x,y,z): {dimensions}")
        print(f"Center point: {center}")
        print(f"Bounding box:")
        print(f"  Min (x,y,z): {min_coords}")
        print(f"  Max (x,y,z): {max_coords}")
        print(f"Number of vertices: {len(vertices)}")


if __name__ == "__main__":
    analyze_obj("3d_models/bunny.obj")
