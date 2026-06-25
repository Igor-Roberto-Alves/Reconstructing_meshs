import os
import io
import struct
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Point Cloud Parsers ───────────────────────────────────────────────────────

def _parse_xyz(text):
    """Parse whitespace/comma-separated x y z [nx ny nz] text files."""
    points, normals = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            row = [float(x) for x in parts[:6]]
        except ValueError:
            continue
        points.append(row[:3])
        normals.append(row[3:6] if len(row) >= 6 else None)

    pts = np.array(points, dtype=float)
    has_normals = all(n is not None for n in normals)
    nrm = np.array(normals, dtype=float) if has_normals else None
    return pts, nrm


def _parse_ply(data):
    """Parse ASCII or binary-little-endian PLY files."""
    # Find header end
    header_end = data.find(b"end_header")
    if header_end == -1:
        raise ValueError("Invalid PLY: missing end_header")
    header = data[:header_end].decode("utf-8", errors="replace")
    body = data[header_end + len("end_header"):]
    while body and body[0:1] in (b"\n", b"\r"):
        body = body[1:]

    # Parse header
    is_binary = "binary_little_endian" in header
    vertex_count = 0
    props = []
    in_vertex = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and "vertex" not in line:
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            props.append((parts[-1], parts[1]))  # (name, type)

    prop_names = [p[0] for p in props]
    prop_types = {p[0]: p[1] for p in props}

    # Map PLY types to struct/numpy
    type_map = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8), "float64": ("d", 8),
        "int": ("i", 4), "int32": ("i", 4),
        "uint": ("I", 4), "uint32": ("I", 4),
        "short": ("h", 2), "int16": ("h", 2),
        "uchar": ("B", 1), "uint8": ("B", 1),
    }

    if is_binary:
        fmt = "<" + "".join(type_map[prop_types[n]][0] for n in prop_names)
        row_size = struct.calcsize(fmt)
        rows = []
        for i in range(vertex_count):
            chunk = body[i * row_size:(i + 1) * row_size]
            rows.append(struct.unpack(fmt, chunk))
        arr = np.array(rows, dtype=float)
    else:
        lines = body.decode("utf-8", errors="replace").splitlines()
        rows = []
        for line in lines[:vertex_count]:
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
        arr = np.array(rows, dtype=float)

    def _col(name):
        if name in prop_names:
            return arr[:, prop_names.index(name)]
        return None

    x, y, z = _col("x"), _col("y"), _col("z")
    if x is None or y is None or z is None:
        raise ValueError("PLY file must have x, y, z vertex properties")
    pts = np.stack([x, y, z], axis=1)

    nx, ny, nz = _col("nx"), _col("ny"), _col("nz")
    nrm = np.stack([nx, ny, nz], axis=1) if (nx is not None and ny is not None and nz is not None) else None
    return pts, nrm


def parse_pointcloud(file_bytes, filename):
    """Dispatch to the right parser based on file extension."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".ply":
        return _parse_ply(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        return _parse_xyz(text)


# ── Mesh → JSON ───────────────────────────────────────────────────────────────

def _mesh_to_json(verts, faces):
    return {
        "vertices": verts.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "vertex_count": len(verts),
        "face_count": len(faces),
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    """
    Accepts a multipart/form-data POST with:
      - file      : point cloud file (.xyz, .txt, .pts, .ply, .csv)
      - algorithm : "bpa", "poisson", or "both" (default "both")
      - bpa_radius: float (default 0.0 = auto-estimate)
      - poisson_depth: int (default 6)
      - max_points: int (default 5000, for performance)

    Returns JSON with bpa and/or poisson mesh data.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    algorithm = request.form.get("algorithm", "both")
    bpa_radius = float(request.form.get("bpa_radius", 0.0))
    poisson_depth = int(request.form.get("poisson_depth", 6))
    max_points = int(request.form.get("max_points", 5000))

    try:
        raw = f.read()
        points, normals_raw = parse_pointcloud(raw, f.filename)
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 400

    if len(points) < 4:
        return jsonify({"error": "Need at least 4 points"}), 400

    # Downsample if too many points
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = points[idx]
        normals_raw = normals_raw[idx] if normals_raw is not None else None

    # Estimate normals if missing
    if normals_raw is None:
        from ballpivot import estimate_normals
        normals = estimate_normals(points)
    else:
        normals = normals_raw

    # Auto-estimate ball radius as 2× median nearest-neighbor distance
    if bpa_radius <= 0:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=2)
        bpa_radius = float(np.median(dists[:, 1]) * 2.5)

    result = {
        "point_count": len(points),
        "bpa_radius_used": bpa_radius,
    }

    if algorithm in ("bpa", "both"):
        try:
            from ballpivot import reconstruct_bpa
            verts, faces = reconstruct_bpa(points, normals, radius=bpa_radius)
            result["bpa"] = _mesh_to_json(verts, faces)
            result["bpa"]["radius"] = bpa_radius
        except Exception:
            result["bpa"] = {"error": traceback.format_exc()}

    if algorithm in ("poisson", "both"):
        try:
            from zpoisson import reconstruct_poisson
            verts, faces = reconstruct_poisson(points, normals, max_depth=poisson_depth)
            result["poisson"] = _mesh_to_json(verts, faces)
            result["poisson"]["depth"] = poisson_depth
        except Exception:
            result["poisson"] = {"error": traceback.format_exc()}

    return jsonify(result)


@app.route("/reconstruct/obj", methods=["POST"])
def reconstruct_obj():
    """Same as /reconstruct but returns OBJ file for download."""
    algorithm = request.form.get("algorithm", "bpa")
    resp = reconstruct()
    if resp.status_code != 200:
        return resp

    data = resp.get_json()
    mesh = data.get(algorithm, {})
    if "error" in mesh:
        return jsonify({"error": mesh["error"]}), 500

    verts_flat = np.array(mesh["vertices"]).reshape(-1, 3)
    faces_flat = np.array(mesh["faces"]).reshape(-1, 3)

    if algorithm == "bpa":
        from ballpivot import export_obj
    else:
        from zpoisson import export_obj

    obj_str = export_obj(verts_flat, faces_flat, name=algorithm)
    from flask import Response
    return Response(obj_str, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{algorithm}_mesh.obj"'})


if __name__ == "__main__":
    print("Starting 3D Reconstruction Server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
