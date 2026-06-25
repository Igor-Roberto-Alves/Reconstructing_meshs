import numpy as np
import os
import sys
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter, laplace, map_coordinates
from scipy.sparse.linalg import cg, LinearOperator

# Ensure octree.py is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octree import Octree


def reconstruct_poisson(points, normals, max_depth=6):
    """
    Screened Poisson Surface Reconstruction (grid-based approximation).

    Solves  ∇·∇χ = ∇·V  where V is the smoothed oriented normal field,
    then extracts the iso-surface with Marching Cubes.

    Parameters
    ----------
    points    : (N, 3) array of point coordinates
    normals   : (N, 3) array of oriented unit normals
    max_depth : octree depth; grid resolution = 2^max_depth
                recommended range: 5 (fast) to 7 (fine detail)

    Returns
    -------
    verts : (V, 3) ndarray of mesh vertices
    faces : (F, 3) ndarray of triangle indices
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    # Normalize normals to unit length (skip degenerate)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms

    grid_res = 2 ** max_depth

    print(f"  Building Octree (depth={max_depth}, grid={grid_res}^3)...")
    tree = Octree(points, max_depth=max_depth)
    tree.insert_cloud(points, normals)
    leaves = tree.get_leaves()

    min_pt = tree.center - (tree.size / 2.0)
    cell = tree.size / grid_res

    # --- 1. Splat oriented normals onto the grid (V field) ---
    print(f"  Splatting {len(leaves)} octree leaves onto vector field...")
    V = np.zeros((grid_res, grid_res, grid_res, 3))
    for leaf in leaves:
        idx = ((leaf.center - min_pt) / tree.size * grid_res).astype(int)
        ix, iy, iz = np.clip(idx, 0, grid_res - 1)
        # Weight by the number of points in this leaf for smoother splatting
        V[ix, iy, iz] += leaf.avg_normal * leaf.weight

    # Smooth V to reduce quantization artifacts
    sigma = max(0.5, grid_res / 64.0)
    for ch in range(3):
        V[:, :, :, ch] = gaussian_filter(V[:, :, :, ch], sigma=sigma)

    # --- 2. Compute divergence  div(V) = dVx/dx + dVy/dy + dVz/dz ---
    print("  Computing divergence...")
    div_V = np.zeros((grid_res, grid_res, grid_res))
    div_V[1:-1, :, :] += (V[2:, :, :, 0] - V[:-2, :, :, 0]) / (2.0 * cell)
    div_V[:, 1:-1, :] += (V[:, 2:, :, 1] - V[:, :-2, :, 1]) / (2.0 * cell)
    div_V[:, :, 1:-1] += (V[:, :, 2:, 2] - V[:, :, :-2, 2]) / (2.0 * cell)

    # --- 3. Solve Poisson equation  ∇²χ = div(V)  via Conjugate Gradients ---
    print("  Solving ∇²χ = div(V) via Conjugate Gradients...")
    N = grid_res ** 3
    rhs = -div_V.flatten()

    # Tikhonov-regularized discrete Laplacian so CG always converges
    epsilon = 1e-4

    def laplacian_op(x_flat):
        x3 = x_flat.reshape((grid_res, grid_res, grid_res))
        lap = -laplace(x3) / (cell ** 2) + epsilon * x3
        return lap.flatten()

    A = LinearOperator((N, N), matvec=laplacian_op)
    chi_flat, code = cg(A, rhs, rtol=1e-5, maxiter=500)
    if code == 0:
        print("  CG converged.")
    else:
        print(f"  CG stopped early (code={code}); result may be approximate.")

    chi = chi_flat.reshape((grid_res, grid_res, grid_res))

    # --- 4. Determine iso-value using the Kazhdan method ---
    # Project original points onto the indicator function and take the mean
    coords = (points - min_pt) / cell
    coords = np.clip(coords, 0, grid_res - 1)
    iso = float(np.mean(map_coordinates(chi, coords.T, order=1)))
    print(f"  Iso-value: {iso:.5f}")

    # --- 5. Extract mesh with Marching Cubes ---
    print("  Running Marching Cubes...")
    spacing = (cell, cell, cell)
    try:
        verts, faces, _, _ = marching_cubes(chi, level=iso, spacing=spacing)
    except ValueError as e:
        raise RuntimeError(f"Marching Cubes failed: {e}. Try a different max_depth.")

    verts += min_pt  # shift back to world coordinates
    return verts, faces


def export_obj(verts, faces, name="mesh"):
    """Returns an OBJ file string from vertex and face arrays."""
    lines = [f"# Poisson Mesh: {name}", f"# {len(verts)} vertices, {len(faces)} faces", ""]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append("")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    return "\n".join(lines)


# --- Standalone test ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("Generating point cloud (full sphere)...")
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, np.pi, 3000)
    theta = rng.uniform(0, 2 * np.pi, 3000)
    pts = np.vstack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)]).T
    nrm = pts.copy()

    verts, faces = reconstruct_poisson(pts, nrm, max_depth=5)
    print(f"Done. Vertices: {len(verts)}, Triangles: {len(faces)}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.6,
                            facecolor='steelblue', edgecolor='k', linewidths=0.1)
    ax.add_collection3d(mesh)
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    ax.set_title(f"Poisson Reconstruction — {len(faces)} triangles")
    plt.tight_layout()
    plt.show()
