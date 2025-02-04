import taichi as ti
import numpy as np
from PIL import Image


@ti.data_oriented
class Renderer():
    def __init__(self,
                 image_size: tuple[int],
                 probe_state_buffer: ti.ndarray, # w, h, stroke
                 mouse_state_buffer: ti.ndarray, # x, y, pressed

    ):
        self.image_size = image_size
        self.probe_state_buffer = probe_state_buffer
        self.mouse_state_buffer = mouse_state_buffer

        self.generate_data_fields()

    def generate_data_fields(self):
        self.image = ti.Vector.field(3, dtype=float, shape=self.image_size)  # RGB field for colors
        self.bg = ti.Vector.field(3, dtype=float, shape=self.image_size)  # Background image
        self.bg_np = np.zeros(self.image_size + (3,))  # Background image as numpy array
        self.probe = ti.Vector.field(3, dtype=float, shape=self.image_size)  # probe image

        # Mouse and probe states
        self.mouse_state = ti.field(dtype=int, shape=(3))  # x, y, pressed
        self.probe_state = ti.field(dtype=int, shape=(3))  # w, h, stroke

        # Initialize
        self.image.fill(0)

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize(self.image_size)
        img = np.array(img, dtype=np.float32) / 255
        self.bg.from_numpy(img)
        self.bg_np = img


    def draw(self):
        self.read_buffers()
        self.draw_kernel()

    @ti.kernel
    def draw_kernel(self):
        self.draw_image()
        self.draw_probe()

    def read_buffers(self):
        self.mouse_state.from_numpy(self.mouse_state_buffer)
        self.probe_state.from_numpy(self.probe_state_buffer)

    @ti.func
    def draw_image(self):
        """Render the image and the probe over it."""
        center_x, center_y, mouse_pressed = self.mouse_state[0], self.mouse_state[1], self.mouse_state[2]
        probe_w, probe_h, probe_stroke = self.probe_state[0], self.probe_state[1], self.probe_state[2]
        color = ti.Vector([1.0, 0.0, 0.0]) if mouse_pressed else ti.Vector([1.0, 1.0, 1.0])  # Red or white

        for i, j in self.image:
            dist_x = abs(i - center_x)
            dist_y = abs(j - center_y)
            if dist_x <= probe_w//2 and dist_y <= probe_h//2 and (dist_x >= probe_w//2 - probe_stroke or dist_y >= probe_h//2 - probe_stroke):
                    self.image[i, j] = color  # Probe color
            else:
                self.image[i, j] = self.bg[i, j]  # Background image

    @ti.func
    def draw_probe(self):
        """Render the probe contents."""
        center_x, center_y = self.mouse_state[0], self.mouse_state[1]
        probe_w, probe_h = self.probe_state[0], self.probe_state[1]

        for i, j in self.probe:
            if i < probe_w and j < probe_h:
                # get bg under square
                bg_x = i + center_x - (probe_w // 2)
                bg_y = j + center_y - (probe_h // 2)
                self.probe[i, j] = self.bg[bg_x, bg_y]
            else:
                self.probe[i, j] = ti.Vector([1.0, 1.0, 1.0]) # White


    def get_probe_matrix(self):
        """Get the probe matrix from the background image."""
        x, y = self.mouse_state[0], self.mouse_state[1]
        probe_w, probe_h = self.probe_state[0], self.probe_state[1]
        x_from = max(x - probe_w//2, 0)
        y_from = max(y - probe_h//2, 0)
        probe = self.bg_np[x_from : x_from + probe_w, y_from : y_from + probe_h]
        return probe
