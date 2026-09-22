import numpy as np

def compute_edge_cost(v0_idx, v1_idx, verts, tris, v_to_t, t_normals, mode="none"):
    """
    Computes the collapse cost of an edge based on geometric length and quality flags.
    """
    p0 = verts[v0_idx]
    p1 = verts[v1_idx]
    
    # Base Cost: Geometric Euclidean Distance (Shorter edges are cheaper)
    base_dist = np.linalg.norm(p0 - p1)
    
    # Find triangles shared by this edge
    shared_tris = v_to_t[v0_idx].intersection(v_to_t[v1_idx])
    
    # ----------------------------------------------------
    # FLAG 1: PRESERVE BOUNDARIES
    # ----------------------------------------------------
    if mode == "preserve_boundaries":
        # An edge is an open boundary if it is shared by exactly 1 triangle
        if len(shared_tris) == 1:
            # Heavily penalize boundary collapses to freeze the outer border shape
            return base_dist * 15.0
            
    # ----------------------------------------------------
    # FLAG 2: PRESERVE VOLUME
    # ----------------------------------------------------
    if mode == "preserve_volume":
        # Find all neighbor triangles that touch either vertex but are NOT shared
        all_tris = v_to_t[v0_idx].union(v_to_t[v1_idx])
        surrounding_tris = all_tris.difference(shared_tris)
        
        if len(surrounding_tris) > 1:
            # Measure the maximum normal deviation (curvature) around the edge
            normals = [t_normals[t_id] for t_id in surrounding_tris if t_id in t_normals]
            if len(normals) > 1:
                # Dot product variation metric
                min_dot = 1.0
                for i in range(len(normals)):
                    for j in range(i + 1, len(normals)):
                        dot = np.dot(normals[i], normals[j])
                        min_dot = min(min_dot, dot)
                
                # Curvature factor goes up as normals diverge (faces are at sharp angles)
                curvature_factor = 1.0 + (1.0 - np.clip(min_dot, -1.0, 1.0)) * 5.0
                return base_dist * curvature_factor

    # Default / Mode "none"
    return base_dist

def compute_triangle_normal(p0, p1, p2):
    """Calculates the normalized surface normal vector of a triangle."""
    v0 = p1 - p0
    v1 = p2 - p0
    n = np.cross(v0, v1)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-8 else np.zeros(3)

def decimate_mesh(vertices, triangles, target_reduction=0.5, mode="none"):
    """
    Decimates a triangular mesh targeting a precise reduction percentage.
    
    Parameters:
    - vertices: Nx3 array-like of floats
    - triangles: Mx3 array-like of ints
    - target_reduction: float (e.g. 0.40 removes 40% of faces)
    - mode: string flag -> 'none', 'preserve_boundaries', or 'preserve_volume'
    """
    assert mode in ["none", "preserve_boundaries", "preserve_volume"], "Invalid metric flag."
    
    verts = np.array(vertices, dtype=float)
    tris = {i: list(t) for i, t in enumerate(triangles)}
    
    # Build vertex-to-triangle lookup map
    v_to_t = {i: set() for i in range(len(verts))}
    for t_id, t in tris.items():
        for v in t:
            v_to_t[v].add(t_id)

    # Compute initial triangle surface normals
    t_normals = {}
    for t_id, t in tris.items():
        t_normals[t_id] = compute_triangle_normal(verts[t[0]], verts[t[1]], verts[t[2]])

    initial_face_count = len(tris)
    target_face_count = int(initial_face_count * (1.0 - target_reduction))
    
    # Gather all unique edges
    edges = set()
    for t in tris.values():
        edges.update([
            (min(t[0], t[1]), max(t[0], t[1])),
            (min(t[1], t[2]), max(t[1], t[2])),
            (min(t[2], t[0]), max(t[2], t[0]))
        ])
        
    # Queue up initial edge costs
    edge_queue = []
    for v0, v1 in edges:
        cost = compute_edge_cost(v0, v1, verts, tris, v_to_t, t_normals, mode)
        edge_queue.append([cost, v0, v1])
    edge_queue.sort(key=lambda x: x[0])

    # Iterative processing loop
    while len(tris) > target_face_count and edge_queue:
        _, v0, v1 = edge_queue.pop(0)
        
        # Guard against already processed/deleted vertices
        if v0 not in v_to_t or v1 not in v_to_t:
            continue
            
        # Merge target v1 into v0 (Midpoint position placement)
        new_pos = (verts[v0] + verts[v1]) / 2.0
        verts[v0] = new_pos
        
        shared_triangles = v_to_t[v0].intersection(v_to_t[v1])
        all_affected_triangles = v_to_t[v0].union(v_to_t[v1])
        to_delete = set(shared_triangles)
        
        # Remap references from v1 to v0
        for t_id in all_affected_triangles:
            if t_id in to_delete:
                continue
                
            t = tris[t_id]
            if t[0] == v1: t[0] = v0
            if t[1] == v1: t[1] = v0
            if t[2] == v1: t[2] = v0
            
            # Identify flat/degenerate faces
            if t[0] == t[1] or t[1] == t[2] or t[2] == t[0]:
                to_delete.add(t_id)
            else:
                v_to_t[v0].add(t_id)
                # Recalculate face normal since geometry shifted
                t_normals[t_id] = compute_triangle_normal(verts[t[0]], verts[t[1]], verts[t[2]])

        # Clean up deleted indices from mappings
        for t_id in to_delete:
            if t_id in tris:
                for v in tris[t_id]:
                    if v in v_to_t:
                        v_to_t[v].discard(t_id)
                del tris[t_id]
                if t_id in t_normals:
                    del t_normals[t_id]
                    
        del v_to_t[v1] # Retire v1 permanently

        # Update neighboring costs local to the modified region
        neighbors = set()
        for t_id in v_to_t[v0]:
            neighbors.update(tris[t_id])
        neighbors.discard(v0)
        
        for n in neighbors:
            cost = compute_edge_cost(v0, n, verts, tris, v_to_t, t_normals, mode)
            edge_queue.append([cost, min(v0, n), max(v0, n)])
            
        # Re-sort list based on updated local structural evaluations
        edge_queue.sort(key=lambda x: x[0])

    # Re-map sparse indices to build a packed vertex index buffer
    remaining_v_indices = sorted(list(v_to_t.keys()))
    index_mapping = {old: new for new, old in enumerate(remaining_v_indices)}
    
    final_vertices = verts[remaining_v_indices]
    final_triangles = []
    for t in tris.values():
        final_triangles.append([index_mapping[t[0]], index_mapping[t[1]], index_mapping[t[2]]])
        
    return final_vertices, np.array(final_triangles)

# --- Verification Test Rig ---
if __name__ == "__main__":
    # Test Geometry: Open Plane Grid (Contains boundary edges and interior volume changes)
    plane_verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.5], [1.0, 1.0, 1.0], [2.0, 1.0, 0.5],  # Center ridge line
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]
    ])
    
    plane_tris = np.array([, [0, 4, 3], [1, 2, 5], [1, 5, 4],
, [3, 7, 6], [4, 5, 8], [4, 8, 7]
    ])
    
    print(f"Original Mesh Config: {len(plane_verts)} Vertices, {len(plane_tris)} Triangles.\n")
    
    for flag in ["none", "preserve_boundaries", "preserve_volume"]:
        v_out, t_out = decimate_mesh(plane_verts, plane_tris, target_reduction=0.5, mode=flag)
        print(f"[{flag.upper()}] Decimated down to -> {len(t_out)} Triangles")
