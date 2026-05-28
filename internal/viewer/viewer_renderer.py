import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh

SH_C0 = 0.28209479177387814  # zeroth-order SH basis constant


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0) / 4
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 4


def gradient_map(image):
    sobel_x = _SOBEL_X.to(image.device)
    sobel_y = _SOBEL_Y.to(image.device)
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    return torch.sqrt(grad_x ** 2 + grad_y ** 2).norm(dim=0, keepdim=True)


class _CudaTimer:
    """Synchronised GPU wall-clock timer used for per-section profiling."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end   = torch.cuda.Event(enable_timing=True)
        self.ms    = 0.0

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *_):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)


# ──────────────────────────────────────────────────────────────────────────────
# ViewerRenderer
# ──────────────────────────────────────────────────────────────────────────────

class ViewerRenderer:
    def __init__(self,
                 gaussian_model,
                 background_color,
                 do_initialize: bool = True,
                 # Frustum culling
                 octree: dict | None = None,
                 culling_enabled: bool = True,
                 # Per-frame profiling
                 profiling_enabled: bool = True,
                 profiling_warmup: int = 5,
                 profiling_print_every: int = 30):
        """
        Parameters
        ----------
        octree
            Dict with keys node_aabbs / node_offsets / flat_indices loaded by
            Viewer._load_octree_idx().  None disables frustum culling.
        culling_enabled
            Set False to bypass culling even when an octree is present.
        profiling_enabled
            Print GPU timing (A:sh / B:raster / C:post) every
            *profiling_print_every* frames after *profiling_warmup* warmup frames.
        """
        super().__init__()
        self.gaussian_model   = gaussian_model
        self.background_color = background_color
        self.clm_colors       = torch.tensor(plt.cm.get_cmap("turbo").colors, device="cuda")

        # Frustum-culling state
        self.octree             = octree
        self.culling_enabled    = culling_enabled
        self._spatially_ordered = False

        # Profiling state
        self.profiling_enabled     = profiling_enabled
        self._prof_warmup          = profiling_warmup
        self._prof_print_every     = profiling_print_every
        self._prof_frame_count     = 0
        self._prof_history: dict   = collections.defaultdict(list)

        if do_initialize:
            self.update_pc_features()

        self._log_startup()

    def _log_startup(self):
        if self.culling_enabled and self.octree is not None:
            L = len(self.octree["node_aabbs"])
            print(f"[viewer] Frustum culling : enabled — {L:,} leaf nodes")
        elif not self.culling_enabled:
            print("[viewer] Frustum culling : disabled via --no-culling")
        else:
            print("[viewer] Frustum culling : no index found — run with --build-index to enable")

        if self.profiling_enabled:
            print(f"[viewer] Profiling       : enabled "
                  f"(warmup {self._prof_warmup} frames, "
                  f"print every {self._prof_print_every} frames)")
        else:
            print("[viewer] Profiling       : disabled via --no-profiling")

    # ── public helpers ────────────────────────────────────────────────────────

    def update_pc_features(self):
        self.means3D   = self.gaussian_model.get_xyz
        self.all_ids   = torch.ones(self.means3D.shape[0], dtype=torch.bool,
                                    device=self.means3D.device)
        self.means2D   = torch.zeros_like(self.means3D)
        self.opacity   = self.gaussian_model.get_opacity
        self.scales    = self.gaussian_model.get_scaling
        self.rotations = self.gaussian_model.get_rotation
        self.shs       = self.gaussian_model.get_features

        if self.octree is not None and not self._spatially_ordered:
            # Reorder all splat tensors to match octree flat_indices leaf order.
            # After this, leaf j maps to contiguous indices [node_offsets[j], node_offsets[j+1]),
            # enabling direct slice gather instead of scattered bool-mask gather.
            # Done one attribute at a time so peak VRAM stays at 2× per-attribute, not 2× total.
            perm = torch.from_numpy(
                self.octree["flat_indices"].astype(np.int64)
            ).to(self.means3D.device)
            for attr in ('_xyz', '_opacity', '_scaling', '_rotation',
                         '_features_dc', '_features_rest'):
                old = getattr(self.gaussian_model, attr)
                setattr(self.gaussian_model, attr, old[perm].contiguous())
                del old
            del perm
            self._spatially_ordered = True
            self.means3D   = self.gaussian_model.get_xyz
            self.means2D   = torch.zeros_like(self.means3D)
            self.opacity   = self.gaussian_model.get_opacity
            self.scales    = self.gaussian_model.get_scaling
            self.rotations = self.gaussian_model.get_rotation
            self.shs       = self.gaussian_model.get_features
            print(f"[viewer] Spatial reorder : {self.means3D.shape[0]:,} splats sorted by octree leaf")

    def disk_kernel(self, opacity):
        return torch.exp(-0.5 * 100 * torch.clamp(opacity - 0.5, min=0) ** 2)

    def color_map(self, map):
        if map.min() == map.max():
            idx = torch.zeros_like(map, device=map.device).round().long().squeeze()
        else:
            map = (map - map.min()) / (map.max() - map.min())
            idx = (map * 255).round().long().squeeze()
        return self.clm_colors[idx].permute(2, 0, 1)

    # ── frustum culling ───────────────────────────────────────────────────────

    def _apply_frustum_cull(self,
                             is_in_box: torch.Tensor,
                             viewpoint_camera):
        """
        Narrow *is_in_box* to splats whose octree leaf intersects the view frustum.

        Returns (bool_mask, vis_ranges) where vis_ranges is a list of (start, end)
        integer pairs when splats are spatially ordered and no crop box is active.
        vis_ranges enables a fast contiguous-slice gather in render_viewer(); it is
        None whenever a fallback bool-mask gather is required (crop box active, or
        model not yet spatially reordered).

        Uses 5 planes (left / right / top / bottom / near).  The far plane is
        deliberately omitted: the GS rasterizer does not hard-clip at zfar, so
        scenes with objects far away would be incorrectly culled by the projection
        matrix's zfar = 100 m.

        Matrix convention: p_clip = p_world @ full_proj_transform  (row-vector).
        Planes are extracted from the columns of M (Gribb-Hartmann method).
        """
        M        = viewpoint_camera.full_proj_transform.detach().cpu().numpy()  # [4,4]
        planes   = np.stack([
            M[:, 0] + M[:, 3],   # left
            M[:, 3] - M[:, 0],   # right
            M[:, 1] + M[:, 3],   # bottom
            M[:, 3] - M[:, 1],   # top
            M[:, 2],              # near (camera-Z >= znear)
        ], axis=0)                # [5, 4]

        normals = planes[:, :3]   # [5, 3]
        d_vals  = planes[:, 3]    # [5]

        node_aabbs   = self.octree["node_aabbs"]    # float32 [L, 6]
        node_offsets = self.octree["node_offsets"]  # int64   [L+1]

        aabb_min = node_aabbs[:, :3]   # [L, 3]
        aabb_max = node_aabbs[:, 3:]   # [L, 3]

        # Gribb-Hartmann p-vertex test, vectorised over all L nodes.
        pos_mask = normals[:, np.newaxis, :] >= 0
        p_vert   = np.where(pos_mask, aabb_max[np.newaxis], aabb_min[np.newaxis])  # [5,L,3]
        dots     = (p_vert * normals[:, np.newaxis, :]).sum(axis=2) + d_vals[:, np.newaxis]
        node_vis = (dots >= 0).all(axis=0)   # [L]

        N_total       = is_in_box.shape[0]
        vis_cpu       = np.zeros(N_total, dtype=np.bool_)
        visible_nodes = np.where(node_vis)[0]
        vis_ranges    = None

        if len(visible_nodes) > 0:
            starts = node_offsets[visible_nodes]
            ends   = node_offsets[visible_nodes + 1]

            if self._spatially_ordered:
                # Leaf j → contiguous [node_offsets[j], node_offsets[j+1]): use
                # direct slice assignment (numpy memset, faster than flat_indices scatter).
                for s, e in zip(starts, ends):
                    vis_cpu[s:e] = True
                # Expose ranges for contiguous GPU gather only when no crop box is active.
                if is_in_box is self.all_ids:
                    vis_ranges = list(zip(starts.tolist(), ends.tolist()))
            else:
                flat_indices = self.octree["flat_indices"]
                all_idx = np.concatenate([flat_indices[s:e] for s, e in zip(starts, ends)])
                vis_cpu[all_idx] = True

        bool_mask = is_in_box & torch.from_numpy(vis_cpu).to(is_in_box.device)
        return bool_mask, vis_ranges

    # ── profiling ─────────────────────────────────────────────────────────────

    def _record_profile(self, n_vis: int, sh_ms: float,
                        raster_ms: float, post_ms: float):
        self._prof_frame_count += 1
        if self._prof_frame_count <= self._prof_warmup:
            return

        h = self._prof_history
        h["sh"].append(sh_ms)
        h["raster"].append(raster_ms)
        h["post"].append(post_ms)
        total = sh_ms + raster_ms + post_ms
        h["total"].append(total)

        if len(h["total"]) % self._prof_print_every != 0:
            return

        W   = self._prof_print_every
        avg = lambda k: float(np.mean(h[k][-W:]))
        sh_a, rast_a, post_a, tot_a = avg("sh"), avg("raster"), avg("post"), avg("total")
        fps  = 1000.0 / tot_a if tot_a > 0 else 0.0
        parts = {"A:sh": sh_a, "B:raster": rast_a, "C:post": post_a}
        bneck = max(parts, key=parts.get)
        pct   = 100.0 * parts[bneck] / tot_a
        print(
            f"[PROFILE] n={n_vis:,}  "
            f"A:sh={sh_a:.1f}ms  B:raster={rast_a:.1f}ms  C:post={post_a:.1f}ms  "
            f"total={tot_a:.1f}ms  {fps:.1f}fps  "
            f"bottleneck→{bneck}({pct:.0f}%)"
        )

    # ── main render path ──────────────────────────────────────────────────────

    def render_viewer(self,
                      viewpoint_camera,
                      active_sh_degree,
                      scaling_modifier,
                      depth_ratio,
                      bg_color: torch.Tensor,
                      sparsity: int = 1,
                      opacity_threshold: float = 0.0,
                      show_ptc: bool = False,
                      show_disk: bool = False,
                      point_size: float = 0.001,
                      valid_range=None,
                      compute_post: bool = True):
        """
        Render the scene.  bg_color must be on GPU.

        Profiling breakdown (printed when profiling_enabled=True):
          A:sh     — SH evaluation (pre-computed here for accurate timing)
          B:raster — CUDA sort + alpha-composite kernel
          C:post   — normal / depth / distortion maps
        """
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # ── spatial crop box ──────────────────────────────────────────────────
        if valid_range is not None:
            is_in_box = (
                (valid_range[0][0] <= self.means3D[:, 0]) & (self.means3D[:, 0] <= valid_range[0][1]) &
                (valid_range[1][0] <= self.means3D[:, 1]) & (self.means3D[:, 1] <= valid_range[1][1]) &
                (valid_range[2][0] <= self.means3D[:, 2]) & (self.means3D[:, 2] <= valid_range[2][1])
            )
        else:
            is_in_box = self.all_ids

        # ── frustum culling (CPU octree walk + GPU mask) ──────────────────────
        vis_ranges = None
        if self.culling_enabled and self.octree is not None:
            is_in_box, vis_ranges = self._apply_frustum_cull(is_in_box, viewpoint_camera)

        # ── opacity threshold (GPU, removes near-transparent splats) ──────────
        if opacity_threshold > 0.0:
            is_in_box = is_in_box & (self.opacity[:, 0] > opacity_threshold)
            vis_ranges = None  # ranges no longer safe with per-splat opacity filter

        # ── gather visible splats ─────────────────────────────────────────────
        # Fast path (spatially ordered, no crop box, no opacity threshold):
        # contiguous slice-cat per leaf — avoids scattered bool-mask gather.
        if vis_ranges:
            _cat = lambda t: torch.cat([t[s:e:sparsity] for s, e in vis_ranges])
        else:
            _cat = lambda t: t[is_in_box][::sparsity]

        means3D_f = _cat(self.means3D)
        means2D_f = _cat(self.means2D)
        opacity_f = _cat(self.opacity)
        if show_disk:
            opacity_f = self.disk_kernel(opacity_f)
        scales_f = _cat(self.scales)
        if show_ptc:
            scales_f = torch.full(scales_f.shape, point_size * 0.1, device=scales_f.device)
        else:
            scales_f = scaling_modifier * scales_f
        rot_f = _cat(self.rotations)
        shs_f = _cat(self.shs)

        # ── [A] SH evaluation ─────────────────────────────────────────────────
        # SH is always pre-computed here (not inside the rasterizer) so we can
        # time it accurately with profiling_enabled=True.  The result is
        # identical: same eval_sh function, same precision.
        if self.profiling_enabled:
            timer_sh = _CudaTimer()
            timer_sh.__enter__()

        if active_sh_degree > 0:
            dir_vecs = means3D_f - viewpoint_camera.camera_center
            dir_vecs = dir_vecs / (dir_vecs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim   = (active_sh_degree + 1) ** 2
            colors   = eval_sh(active_sh_degree,
                               shs_f.transpose(1, 2)[:, :, :sh_dim],
                               dir_vecs)
            colors   = torch.clamp_min(colors + 0.5, 0.0)
        else:
            colors = torch.clamp_min(SH_C0 * shs_f[:, 0, :] + 0.5, 0.0)

        if self.profiling_enabled:
            timer_sh.__exit__(None, None, None)

        # ── [B] Rasterizer ────────────────────────────────────────────────────
        if self.profiling_enabled:
            timer_raster = _CudaTimer()
            timer_raster.__enter__()

        rendered_image, radii, allmap = rasterizer(
            means3D        = means3D_f,
            means2D        = means2D_f,
            shs            = None,
            colors_precomp = colors,
            opacities      = opacity_f,
            scales         = scales_f,
            rotations      = rot_f,
            cov3D_precomp  = None,
        )

        if self.profiling_enabled:
            timer_raster.__exit__(None, None, None)

        # ── [C] Post-processing ───────────────────────────────────────────────
        if self.profiling_enabled:
            timer_post = _CudaTimer()
            timer_post.__enter__()

        if compute_post:
            render_alpha          = allmap[1:2]
            render_normal         = allmap[2:5]
            render_normal         = (
                render_normal.permute(1, 2, 0) @
                viewpoint_camera.world_view_transform[:3, :3].T
            ).permute(2, 0, 1)
            render_depth_median   = torch.nan_to_num(allmap[5:6], 0, 0)
            render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
            render_dist           = allmap[6:7]
            surf_depth  = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2, 0, 1) * render_alpha.detach()
            render_normal = F.normalize(render_normal, dim=0) * 0.5 + 0.5
            surf_normal   = surf_normal * 0.5 + 0.5
            view_normal   = -F.normalize(allmap[2:5], dim=0) * 0.5 + 0.5

        if self.profiling_enabled:
            timer_post.__exit__(None, None, None)
            n_vis = (sum(e - s for s, e in vis_ranges) if vis_ranges
                     else int(is_in_box.sum())) // sparsity
            self._record_profile(n_vis, timer_sh.ms, timer_raster.ms, timer_post.ms)

        if not compute_post:
            return {"render": rendered_image}

        return {
            "render":          rendered_image,
            "rend_alpha":      self.color_map(render_alpha.unsqueeze(-1)),
            "rend_normal":     render_normal,
            "view_normal":     view_normal,
            "surf_depth":      self.color_map(surf_depth.unsqueeze(-1)),
            "surf_depth_raw":  surf_depth.squeeze(0),   # float32 [H, W] in scene units
            "surf_normal":     surf_normal,
            "rend_dist":       self.color_map(render_dist.unsqueeze(-1)),
        }

    # ── output routing ────────────────────────────────────────────────────────

    # Render types that only need the raw rendered image (no allmap post-processing).
    _POST_FREE = frozenset({"render", "edge"})

    def get_outputs(self,
                    camera,
                    valid_range: tuple = None,
                    split: bool = False,
                    slider: float = 0.5,
                    show_ptc: bool = False,
                    show_disk: bool = False,
                    point_size: float = 0.01,
                    active_sh_degree: int = 3,
                    scaling_modifier: float = 1.,
                    sparsity: int = 1,
                    opacity_threshold: float = 0.0,
                    depth_ratio: float = 0.,
                    render_type: str = "render",
                    render_type1: str = "render",
                    render_type2: str = "render"):

        def get_result(results, rtype):
            if rtype in results:
                return results[rtype]
            if rtype == "curvature":
                return self.color_map(gradient_map(results["surf_normal"]))
            if rtype == "edge":
                return self.color_map(gradient_map(results["render"]))
            return results["render"]

        if split:
            compute_post = not (render_type1 in self._POST_FREE and render_type2 in self._POST_FREE)
        else:
            compute_post = render_type not in self._POST_FREE

        results = self.render_viewer(
            camera, active_sh_degree, scaling_modifier, depth_ratio,
            self.background_color,
            sparsity=sparsity, opacity_threshold=opacity_threshold,
            valid_range=valid_range,
            show_ptc=show_ptc, show_disk=show_disk, point_size=point_size,
            compute_post=compute_post,
        )

        if not split:
            return get_result(results, render_type)

        out = torch.zeros_like(results["render"])
        _, _, H = out.shape
        sp = int(H * slider)
        out[:, :, :sp]  = get_result(results, render_type1)[:, :, :sp]
        out[:, :, sp:]  = get_result(results, render_type2)[:, :, sp:]
        out[:, :, sp]   = torch.ones_like(out[:, :, sp])
        return out
