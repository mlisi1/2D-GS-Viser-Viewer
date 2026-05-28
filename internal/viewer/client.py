import time
import threading
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf
from internal.cameras.cameras import Cameras
from internal.utils.graphics_utils import fov2focal

class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client
        self.render_trigger = threading.Event()
        self.last_move_time = 0
        self.last_camera = None  # store camera information
        self.state = "low"  # low or high render resolution
        self.stop_client = False  # whether stop this thread

        # Encode thread: always sends the latest rendered frame, dropping stale ones.
        # This decouples GPU rendering from CPU JPEG encoding so both run in parallel.
        self._send_slot  = None   # (image_np uint8, jpeg_quality, t_render_start)
        self._send_lock  = threading.Lock()
        self._send_ready = threading.Event()
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

        if viewer.default_camera_position is not None:
            client.camera.position = np.asarray(viewer.default_camera_position)
        if viewer.default_camera_look_at is not None:
            client.camera.look_at = np.asarray(viewer.default_camera_look_at)
        client.camera.up_direction = viewer.up_direction

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()
        if hasattr(viewer, 'render_panel'):
            @viewer.render_panel.preview_button.on_click
            def _(_) -> None:
                with self.client.atomic():
                    self.render_trigger.set()
            @viewer.render_panel.STOP.on_click
            def _(_) -> None:
                with self.client.atomic():
                    self.render_trigger.set()
            @viewer.render_panel.preview_frame_slider.on_update
            def _(_) -> None:
                with self.client.atomic():
                    self.render_trigger.set()

    # ── encode/send thread ────────────────────────────────────────────────────

    def _send_loop(self):
        """Picks up the latest rendered frame and encodes+sends it.

        Runs concurrently with the render loop so JPEG encoding does not stall
        GPU work.  Frames are dropped (overwritten) if the encoder falls behind.
        """
        while True:
            self._send_ready.wait()
            self._send_ready.clear()
            if self.stop_client:
                break
            with self._send_lock:
                slot = self._send_slot
                self._send_slot = None
            if slot is None:
                continue
            image_np, jpeg_quality = slot
            self.client.set_background_image(
                image_np,
                format=self.viewer.image_format,
                jpeg_quality=jpeg_quality,
            )

    # ── camera helpers ────────────────────────────────────────────────────────

    def get_RT(self, wxyz, position):
        R = vtf.SO3(wxyz=wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(position, dtype=torch.float64)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        c2w = torch.matmul(self.viewer.camera_transform, c2w)
        c2w[:3, 1:3] *= -1

        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        return R, T

    def make_camera(self, cam, ptc_mode=False):
        max_res, _ = self.get_render_options()
        image_height = max_res
        image_width = int(image_height * cam.aspect)

        if image_width > max_res:
            image_width = max_res
            image_height = int(image_width / cam.aspect)

        if ptc_mode:
            return image_width, image_height

        fx = torch.tensor([fov2focal(cam.fov, max_res)], dtype=torch.float)
        R, T = self.get_RT(self.client.camera.wxyz, self.client.camera.position)

        return Cameras(
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            fx=fx,
            fy=fx,
            cx=torch.tensor([(image_width // 2)], dtype=torch.int),
            cy=torch.tensor([(image_height // 2)], dtype=torch.int),
            width=torch.tensor([image_width], dtype=torch.int),
            height=torch.tensor([image_height], dtype=torch.int),
            appearance_id=torch.tensor([0], dtype=torch.int),
            normalized_appearance_id=torch.tensor([0.], dtype=torch.float),
            time=torch.tensor([0], dtype=torch.float),
            distortion_params=None,
            camera_type=torch.tensor([0], dtype=torch.int),
        )[0].to_device(self.viewer.device)

    # ── render helpers ────────────────────────────────────────────────────────

    def render_image(self, camera):
        valid_range = None
        if self.viewer.enable_crop.value:
            valid_range = (self.viewer.box_x.value, self.viewer.box_y.value, self.viewer.box_z.value)
        return self.renderer.get_outputs(
            camera,
            valid_range=valid_range,
            split=self.viewer.enable_split.value,
            slider=self.viewer.mode_slider.value,
            show_ptc = self.viewer.enable_ptc.value and (self.viewer.surfel_mode.value == 'ptc'),
            show_disk = self.viewer.enable_ptc.value and (self.viewer.surfel_mode.value == 'disk'),
            point_size=self.viewer.point_size.value,
            active_sh_degree = self.viewer.active_sh_degree_slider.value if hasattr(self.viewer, 'active_sh_degree_slider') else 0,
            scaling_modifier = self.viewer.scale_slider.value,
            sparsity = self.viewer.sparsity_slider.value,
            opacity_threshold = self.viewer.opacity_threshold_slider.value,
            depth_ratio = self.viewer.depth_ratio_slider.value,
            render_type = self.viewer.render_type_name[self.viewer.render_type.value],
            render_type1 = self.viewer.render_type_name[self.viewer.render_type1.value],
            render_type2 = self.viewer.render_type_name[self.viewer.render_type2.value],
        )

    @staticmethod
    def _to_uint8(image: torch.Tensor) -> np.ndarray:
        """Convert a [C, H, W] float GPU tensor to a uint8 HWC numpy array.

        Doing the float→uint8 conversion on the GPU means only 1/4 the data
        crosses the PCIe bus compared with copying a float32 tensor.
        """
        return image.clamp(0., 1.).permute(1, 2, 0).mul(255).byte().cpu().numpy()

    def render_image_from_paths(self, path, camera_params):
        # Construct camera
        R, T = self.get_RT(path["wxyz"], path["position"])
        camera = Cameras(
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            **camera_params
        )[0].to_device(self.viewer.device)

        with torch.no_grad():
            image = self.render_image(camera)

        return self._to_uint8(image)

    # ── camera path playback ──────────────────────────────────────────────────

    def send_camera_path(self, camera_paths, fps=30):
        # calculate default camera information
        max_res, jpeg_quality = self.get_render_options()
        image_height = max_res
        image_width = int(image_height * self.last_camera.aspect)
        if image_width > max_res:
            image_width = max_res
            image_height = int(image_width / self.last_camera.aspect)

        camera_params = {
            'fx': torch.tensor([fov2focal(self.last_camera.fov, max_res)], dtype=torch.float),
            'fy': torch.tensor([fov2focal(self.last_camera.fov, max_res)], dtype=torch.float),
            'cx': torch.tensor([(image_width // 2)], dtype=torch.int),
            'cy': torch.tensor([(image_height // 2)], dtype=torch.int),
            'width': torch.tensor([image_width], dtype=torch.int),
            'height': torch.tensor([image_height], dtype=torch.int),
            'appearance_id': torch.tensor([0], dtype=torch.int),
            'normalized_appearance_id': torch.tensor([0.], dtype=torch.float),
            'time': torch.tensor([0], dtype=torch.float),
            'distortion_params': None,
            'camera_type': torch.tensor([0], dtype=torch.int)
        }

        framenum = self.viewer.render_panel.preview_frame_slider.value
        while framenum < len(camera_paths) and self.viewer.render_panel.STOP.visible:
            with self.client.atomic():
                if self.viewer.render_panel.preview_pause.visible:
                    start = time.time()
                    self.viewer.render_panel.preview_frame_slider.value = framenum
                    image_np = self.render_image_from_paths(camera_paths[framenum], camera_params)
                    framenum += 1
                    self.client.set_background_image(
                        image_np,
                        format=self.viewer.image_format,
                        jpeg_quality=jpeg_quality,
                    )
                    self.render_trigger.set()
                    end = time.time()
                    self.viewer.fps.value = f'{(1 / (end-start)):.1f} frame/sec'
                    self.viewer.gpu_mem.value = self.viewer.get_gpu_memory_usage()
                else:
                    while not self.viewer.render_panel.preview_pause.visible:
                        time.sleep(1/10)

    def set_pre_preview(self):
        self.viewer.render_panel.show_checkbox.value = False
        self.viewer.render_panel.show_splines.value = False
        self.viewer.render_panel.move_checkbox.value = False

    def set_post_preview(self):
        self.viewer.render_panel.play_preview = False
        self.viewer.render_panel.preview_pause.visible = False
        self.viewer.render_panel.STOP.visible = False
        self.viewer.render_panel.preview_button.visible = True
        self.viewer.render_panel.preview_frame_slider.value = 0
        self.viewer.render_panel.show_checkbox.value = True
        self.viewer.render_panel.show_splines.value = True
        self.viewer.render_panel.move_checkbox.value = True

    # ── main render+send path ─────────────────────────────────────────────────

    def render_and_send(self):
        if hasattr(self.viewer, 'render_panel'):
            if self.viewer.render_panel.play_preview:
                if self.viewer.render_panel.preview_cameras is not None:
                    fps = self.viewer.render_panel.preview_cameras['fps']
                    self.set_pre_preview() # hide keyframes / splines / controllers
                    self.send_camera_path(self.viewer.render_panel.preview_cameras['camera_path'], fps)
                    self.set_post_preview() # show hide keyframes / splines / controllers & preview / pause / Stop button to default
                    self.render_trigger.clear()
                    self.render_trigger.set()

        t_start = time.perf_counter()

        # No atomic() here — holding _atomic_lock for the full GPU render (35ms)
        # blocks viser's asyncio event loop, preventing camera updates and image sends.
        with torch.no_grad():
            if hasattr(self.viewer, 'edit_panel') and (self.viewer.edit_panel.show_point_cloud_checkbox.value or self.viewer.edit_panel.mesh is not None):
                image_width, image_height = self.make_camera(self.last_camera, ptc_mode=True)
                image = self.viewer.background_color.unsqueeze(dim=1).unsqueeze(dim=2).expand([3, image_height, image_width])
            else:
                camera = self.make_camera(self.last_camera)
                image = self.render_image(camera)

            # Convert to uint8 on GPU: 4× less data over PCIe than float32.
            image_np = self._to_uint8(image)

        render_ms = 1000.0 * (time.perf_counter() - t_start)
        if render_ms > 0:
            self.viewer.fps.value = f'{1000.0 / render_ms:.1f} frame/sec'
        self.viewer.gpu_mem.value = self.viewer.get_gpu_memory_usage()

        _, jpeg_quality = self.get_render_options()

        # Hand off to encode thread; overwrite any pending (stale) frame.
        with self._send_lock:
            self._send_slot = (image_np, jpeg_quality)
        self._send_ready.set()

    # ── thread lifecycle ──────────────────────────────────────────────────────

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue
                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode
            self.render_trigger.clear()
            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                self.render_trigger.clear()
                self.render_and_send()
        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        self._send_ready.set()  # wake up encode thread so it can exit

    def _destroy(self):
        self._send_thread.join(timeout=2.0)
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None
