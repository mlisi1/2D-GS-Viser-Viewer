#!/usr/bin/env python3
"""
build_index.py — Offline octree builder for frustum culling.

Usage:
    python build_index.py <path_to_ply>

Produces a .idx file (NumPy .npz) alongside the .ply:
    node_aabbs   float32 [num_nodes, 6]      min_xyz + max_xyz per leaf
    node_offsets int64   [num_nodes + 1]     CSR row pointers into flat_indices
    flat_indices int32   [N]                 splat indices grouped by leaf node
"""

import argparse
import os
import sys
import time

import numpy as np
from plyfile import PlyData


MAX_DEPTH      = 7
LEAF_MAX_SPLATS = 170_000   # stop subdividing when node has <= this many splats


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_xyz(ply_path: str) -> np.ndarray:
    """Return float32 [N, 3] XYZ array from a 2DGS PLY."""
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    x = np.asarray(el["x"], dtype=np.float32)
    y = np.asarray(el["y"], dtype=np.float32)
    z = np.asarray(el["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Octree builder
# ──────────────────────────────────────────────────────────────────────────────

def build_octree(
    xyz: np.ndarray,
    max_depth: int = MAX_DEPTH,
    leaf_max: int = LEAF_MAX_SPLATS,
):
    """
    Build an octree over float32 [N, 3] positions.

    Returns
    -------
    node_aabbs   float32 [L, 6]    min_xyz | max_xyz per leaf node
    node_offsets int64   [L + 1]   CSR offsets into flat_indices
    flat_indices int32   [N]       splat indices, one contiguous block per leaf
    """
    N = xyz.shape[0]

    # Tiny AABB padding so boundary splats are never missed.
    global_min = xyz.min(axis=0).astype(np.float64)
    global_max = xyz.max(axis=0).astype(np.float64)
    pad = (global_max - global_min) * 1e-6 + 1e-9
    global_min -= pad
    global_max += pad

    aabb_list: list[np.ndarray] = []
    idx_list:  list[np.ndarray] = []

    # Iterative DFS: stack entries are (indices, aabb_min, aabb_max, depth)
    stack = [(np.arange(N, dtype=np.int32),
              global_min.astype(np.float32),
              global_max.astype(np.float32),
              0)]

    while stack:
        indices, amin, amax, depth = stack.pop()
        n = len(indices)
        if n == 0:
            continue

        # Leaf: max depth reached or node is small enough
        if depth >= max_depth or n <= leaf_max:
            aabb_list.append(np.concatenate([amin, amax]))
            idx_list.append(indices)
            continue

        # Split into 8 octants
        center = ((amin.astype(np.float64) + amax.astype(np.float64)) * 0.5).astype(np.float32)
        pts = xyz[indices]   # [n, 3]

        x_hi = pts[:, 0] >= center[0]
        y_hi = pts[:, 1] >= center[1]
        z_hi = pts[:, 2] >= center[2]

        for bx in range(2):
            mx = x_hi if bx else ~x_hi
            for by in range(2):
                my = y_hi if by else ~y_hi
                for bz in range(2):
                    mz = z_hi if bz else ~z_hi
                    mask = mx & my & mz
                    child_idx = indices[mask]
                    if len(child_idx) == 0:
                        continue

                    child_min = np.array([
                        center[0] if bx else amin[0],
                        center[1] if by else amin[1],
                        center[2] if bz else amin[2],
                    ], dtype=np.float32)
                    child_max = np.array([
                        amax[0] if bx else center[0],
                        amax[1] if by else center[1],
                        amax[2] if bz else center[2],
                    ], dtype=np.float32)

                    stack.append((child_idx, child_min, child_max, depth + 1))

    L = len(aabb_list)
    node_aabbs = np.stack(aabb_list, axis=0).astype(np.float32)   # [L, 6]

    lengths      = np.array([len(a) for a in idx_list], dtype=np.int64)
    node_offsets = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(lengths, out=node_offsets[1:])
    flat_indices = np.concatenate(idx_list).astype(np.int32)

    return node_aabbs, node_offsets, flat_indices


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build an octree frustum-culling index alongside a 2DGS .ply file."
    )
    parser.add_argument("ply_path", help="Path to .ply file")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help=f"Max octree depth (default {MAX_DEPTH})")
    parser.add_argument("--leaf-max", type=int, default=LEAF_MAX_SPLATS,
                        help=f"Stop splitting when node has <= this many splats "
                             f"(default {LEAF_MAX_SPLATS:,})")
    args = parser.parse_args()

    ply_path = args.ply_path
    idx_path = os.path.splitext(ply_path)[0] + ".idx"

    print(f"[build_index] Input:     {ply_path}")
    print(f"[build_index] Output:    {idx_path}")
    print(f"[build_index] max_depth= {args.max_depth}  leaf_max= {args.leaf_max:,}")

    # ── read XYZ ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("[build_index] Reading PLY…")
    xyz = read_xyz(ply_path)
    N   = xyz.shape[0]
    print(f"[build_index]   {N:,} splats   "
          f"({time.perf_counter()-t0:.2f}s)")

    # ── build octree ─────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    print("[build_index] Building octree…")
    node_aabbs, node_offsets, flat_indices = build_octree(
        xyz, max_depth=args.max_depth, leaf_max=args.leaf_max
    )
    dt_build = time.perf_counter() - t1
    print(f"[build_index]   Done in {dt_build:.2f}s")

    # ── verify completeness ───────────────────────────────────────────────────
    assert flat_indices.shape[0] == N, (
        f"BUG: index covers {flat_indices.shape[0]:,} splats but PLY has {N:,}"
    )
    sorted_check = np.sort(flat_indices)
    assert (sorted_check == np.arange(N, dtype=np.int32)).all(), \
        "BUG: flat_indices is not a permutation of [0, N)"

    # ── stats ─────────────────────────────────────────────────────────────────
    L      = len(node_aabbs)
    counts = np.diff(node_offsets)
    print(f"[build_index]   Leaf nodes : {L:,}")
    print(f"[build_index]   Splats/leaf: avg={counts.mean():.0f}  "
          f"min={counts.min():,}  max={counts.max():,}")

    # depth distribution: back-compute from AABB edge lengths
    edge = (node_aabbs[:, 3:] - node_aabbs[:, :3]).max(axis=1)
    root_edge = float(edge.max())
    if root_edge > 0:
        depths = np.round(np.log2(root_edge / np.maximum(edge, 1e-12))).astype(int)
        for d in sorted(set(depths.tolist())):
            cnt = int((depths == d).sum())
            print(f"[build_index]     depth {d}: {cnt:,} leaf nodes")

    # ── save ──────────────────────────────────────────────────────────────────
    # Pass a file object so numpy does not append .npz to the .idx extension.
    t2 = time.perf_counter()
    with open(idx_path, "wb") as fh:
        np.savez_compressed(
            fh,
            node_aabbs   = node_aabbs,
            node_offsets = node_offsets,
            flat_indices = flat_indices,
        )
    sz_mb = os.path.getsize(idx_path) / 1024 / 1024
    print(f"[build_index]   Saved {idx_path}  ({sz_mb:.1f} MB, "
          f"{time.perf_counter()-t2:.2f}s)")
    print(f"[build_index] Total: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
