import numpy as np
from scipy.spatial import KDTree


def estimate_normals(points, k=10):
    """PCA-based normal estimation. Orients normals outward from centroid."""
    tree = KDTree(points)
    normals = np.zeros_like(points)
    centroid = points.mean(axis=0)
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k + 1, len(points)))
        neighbors = points[idx]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]  # eigenvector of smallest eigenvalue = normal
        if np.dot(n, p - centroid) < 0:
            n = -n
        normals[i] = n
    return normals


def _sphere_center(p1, p2, p3, radius, normal_ref):
    """
    Computes the center of a sphere of given radius resting on three points.
    Returns None if the triangle is too large for the ball or degenerate.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    n_len = np.linalg.norm(n)
    if n_len < 1e-10:
        return None
    n = n / n_len
    if np.dot(n, normal_ref) < 0:
        n = -n

    # Solve for circumcenter offset in the triangle's plane
    A = np.array([v1, v2, n])
    b = np.array([0.5 * np.dot(v1, v1), 0.5 * np.dot(v2, v2), 0.0])
    try:
        c_rel = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    r2_circ = np.dot(c_rel, c_rel)
    if r2_circ > radius ** 2:
        return None  # triangle too large for this ball radius

    h = np.sqrt(max(0.0, radius ** 2 - r2_circ))
    return p1 + c_rel + h * n


def _find_seed(points, normals, tree, radius):
    """
    Finds a valid seed triangle: three nearby points where the ball
    fits without enclosing any other point.
    Uses random order to avoid worst-case sequential scanning.
    """
    rng = np.random.default_rng(42)
    order = rng.permutation(len(points))

    for i in order:
        neighbors = tree.query_ball_point(points[i], r=radius * 2.0)
        if len(neighbors) < 3:
            continue
        for ji, j in enumerate(neighbors):
            if j == i:
                continue
            for k in neighbors[ji + 1:]:
                if k == i or k == j:
                    continue
                center = _sphere_center(
                    points[i], points[j], points[k], radius, normals[i]
                )
                if center is None:
                    continue
                # Ball must not enclose any point (seed points are on surface)
                enclosed = tree.query_ball_point(center, r=radius * (1.0 - 1e-5))
                if not enclosed:
                    return [i, j, k], center
    return None, None


def _pivot(i, j, k_opp, points, normals, tree, radius):
    """
    Pivots the ball around edge (i, j), starting from the previous triangle
    with opposite vertex k_opp. Returns the index of the next point touched,
    or None if no valid candidate is found.
    """
    p1, p2 = points[i], points[j]
    mid = (p1 + p2) * 0.5
    edge = p2 - p1
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-10:
        return None
    edge_unit = edge / edge_len

    # Compute average normal for the edge to use as orientation reference
    avg_normal = (normals[i] + normals[j]) * 0.5

    center_prev = _sphere_center(p1, p2, points[k_opp], radius, avg_normal)
    if center_prev is None:
        return None

    # Perpendicular component of the previous center relative to the edge axis
    v_prev = center_prev - mid
    v_prev -= np.dot(v_prev, edge_unit) * edge_unit
    v_prev_len = np.linalg.norm(v_prev)
    if v_prev_len < 1e-10:
        return None
    v_prev_unit = v_prev / v_prev_len

    best_point = None
    best_angle = 2.0 * np.pi

    candidates = tree.query_ball_point(mid, r=radius * 2.0)
    for c in candidates:
        if c == i or c == j or c == k_opp:
            continue
        center_new = _sphere_center(p1, p2, points[c], radius, avg_normal)
        if center_new is None:
            continue
        # Validate: no other point can be strictly inside the new ball
        enclosed = tree.query_ball_point(center_new, r=radius * (1.0 - 1e-5))
        if enclosed:
            continue

        v_new = center_new - mid
        v_new -= np.dot(v_new, edge_unit) * edge_unit
        v_new_len = np.linalg.norm(v_new)
        if v_new_len < 1e-10:
            continue
        v_new_unit = v_new / v_new_len

        cos_a = np.clip(np.dot(v_prev_unit, v_new_unit), -1.0, 1.0)
        angle = np.arccos(cos_a)

        # Determine rotation direction to always advance "forward"
        cross = np.cross(v_prev_unit, v_new_unit)
        if np.dot(cross, edge_unit) < 0:
            angle = 2.0 * np.pi - angle

        if angle < best_angle:
            best_angle = angle
            best_point = c

    return best_point


def reconstruct_bpa(points, normals=None, radius=0.25, max_triangles=50000):
    """
    Ball Pivoting Algorithm surface reconstruction.

    Parameters
    ----------
    points  : (N, 3) array of point coordinates
    normals : (N, 3) array of normals, estimated if None
    radius  : ball radius; should be ~2x the average point spacing
    max_triangles : safety cap to prevent runaway loops

    Returns
    -------
    points : (N, 3) ndarray
    faces  : (M, 3) ndarray of triangle indices, or empty array on failure
    """
    points = np.asarray(points, dtype=float)
    if normals is None:
        normals = estimate_normals(points)
    else:
        normals = np.asarray(normals, dtype=float)

    tree = KDTree(points)
    triangles = []
    # Maps (min_idx, max_idx) -> opposite vertex index, or None when edge is closed
    edge_status = {}
    frontier = []

    seed, _ = _find_seed(points, normals, tree, radius)
    if seed is None:
        return points, np.empty((0, 3), dtype=int)

    triangles.append(seed)
    for a, b, opp in [
        (seed[0], seed[1], seed[2]),
        (seed[1], seed[2], seed[0]),
        (seed[2], seed[0], seed[1]),
    ]:
        key = (min(a, b), max(a, b))
        edge_status[key] = opp
        frontier.append((a, b))

    while frontier and len(triangles) < max_triangles:
        u, v = frontier.pop(0)
        key = (min(u, v), max(u, v))
        if edge_status.get(key) is None:
            continue

        opp = edge_status[key]
        new_pt = _pivot(u, v, opp, points, normals, tree, radius)

        if new_pt is not None:
            # Reversed order [v, u] maintains consistent outward winding
            triangles.append([v, u, new_pt])
            edge_status[key] = None  # close the hinge edge

            for a, b, new_opp in [(v, new_pt, u), (new_pt, u, v)]:
                nkey = (min(a, b), max(a, b))
                if nkey in edge_status:
                    if edge_status[nkey] is not None:
                        edge_status[nkey] = None  # close a loop
                else:
                    edge_status[nkey] = new_opp
                    frontier.append((a, b))

    faces = np.array(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int)
    return points, faces


def export_obj(points, faces, name="mesh"):
    """Returns an OBJ file string from vertex and face arrays."""
    lines = [f"# BPA Mesh: {name}", f"# {len(points)} vertices, {len(faces)} faces", ""]
    for p in points:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    lines.append("")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    return "\n".join(lines)


# --- Standalone test ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("Generating point cloud (upper hemisphere)...")
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, np.pi / 2, 600)
    theta = rng.uniform(0, 2 * np.pi, 600)
    pts = np.vstack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)]).T
    nrm = pts.copy()

    ball_radius = 0.25
    verts, faces = reconstruct_bpa(pts, nrm, radius=ball_radius)
    print(f"Done. Triangles computed: {len(faces)}")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=5, c='crimson', alpha=0.5)
    if len(faces):
        mesh = Poly3DCollection(verts[faces], alpha=0.4,
                                facecolor='dodgerblue', edgecolor='navy', linewidths=0.3)
        ax.add_collection3d(mesh)
    ax.set_title(f"Ball Pivoting — {len(faces)} triangles")
    plt.tight_layout()
    plt.show()
