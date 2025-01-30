from .features import Feature
from .utils import scale_array_exp
import taichi as ti
from ipycanvas import hold_canvas, MultiCanvas
from IPython.display import display
import ipywidgets as widgets
import time
import numpy as np
import signalflow as sf
from PIL import Image
from .utils import samps2mix


class App():
    def __init__(
            self,
            image_size: tuple[int] = (500, 500),
            padding: int = 40,
            fps: int = 60,
            probe_stroke: int = 1,
            ):
        
        self.image_size = image_size
        self.padding = padding
        self.refresh_interval = 1 / fps

        # Global state variables
        self.is_drawing = False
        self.last_draw_time = time.time()
        self.mouse_state = np.array([self.image_size[0]//2, self.image_size[1]//2, 0], dtype=np.int32)  # x, y, pressed
        self.probe_state = np.array([50, 50, probe_stroke], dtype=np.int32)  # w, h, stroke

        # Containers for features, mappers, and synths
        self.features = []
        self.mappers = []
        self.synths = []

        self.create_gui()
        self.create_audio_graph()
        

    def create_gui(self):
        # Create the canvas
        self.canvas = MultiCanvas(
            2,
            width=self.image_size[0]*2 + self.padding*2, 
            height=self.image_size[1] + self.padding*2)
        display(self.canvas)
        
        slider_layout = widgets.Layout(width='400px')
        slider_style = dict(description_width='auto')

        # Create two sliders for controlling the Probe size
        self.probe_w_slider = widgets.IntSlider(
            value=50,  # Initial value
            min=1,  # Minimum size
            max=self.image_size[0],   # Maximum size
            step=1, # Step size
            description='Probe Width',
            style=slider_style,
            layout=slider_layout
        )
        display(self.probe_w_slider)

        self.probe_h_slider = widgets.IntSlider(
            value=50,  # Initial value
            min=1,  # Minimum size
            max=self.image_size[1],   # Maximum size
            step=1, # Step size
            description='Probe Height',
            style=slider_style,
            layout=slider_layout
        )
        display(self.probe_h_slider)

        # Set them to the probe size
        self.probe_state[0], self.probe_state[1] = [self.probe_w_slider.value, self.probe_h_slider.value]

        # Create a toggle button for audio
        self.audio_graph_toggle = widgets.ToggleButton(
            value=False,
            description='Audio',
            disabled=False,
            tooltip='Toggle audio',
            layout=widgets.Layout(width='auto'),
        )
        self.audio_graph_toggle.style.text_color = 'black'
        display(self.audio_graph_toggle)

        # Create a master volume slider
        self.master_volume_slider = widgets.FloatSlider(
            value=0,  # Initial value
            min=-36,  # Minimum value
            max=0,   # Maximum value
            step=0.01, # Step size
            description='Master Volume (dB)',
            style=slider_style,
            layout=slider_layout
        )
        display(self.master_volume_slider)

        # Create two text widgets for displaying the mouse position in px
        self.mouse_x_text = widgets.Text(
            value='',
            description='MouseX:',
            disabled=True
        )
        self.mouse_y_text = widgets.Text(
            value='',
            description='MouseY:',
            disabled=True
        )
        display(self.mouse_x_text)
        display(self.mouse_y_text)

        # Mousing event listeners
        self.canvas.on_mouse_move(lambda x, y: self.mouse_callback(x, y, -1))  # Triggered during mouse movement (keeps track of mouse button state)
        self.canvas.on_mouse_down(lambda x, y: self.mouse_callback(x, y, pressed=2))  # When mouse button pressed
        self.canvas.on_mouse_up(lambda x, y: self.mouse_callback(x, y, pressed=3))  # When mouse button released

        # GUI event listeners
        self.probe_w_slider.observe(self.update_probe_width, names='value')
        self.probe_h_slider.observe(self.update_probe_height, names='value')
        self.audio_graph_toggle.observe(self.toggle_audio, names='value')
        self.master_volume_slider.observe(self.update_master_volume, names='value')

    def create_audio_graph(self):
        self.graph = sf.AudioGraph.get_shared_graph()
        if self.graph is None:
            self.graph = sf.AudioGraph(start=False)
        else:
            self.graph.destroy()
            self.graph = sf.AudioGraph(start=False)

        # DSP switch
        self.dsp_switch_buf = sf.Buffer(1, 1)
        self.dsp_switch_buf.data[0][0] = 0
        self.dsp_switch = sf.BufferPlayer(self.dsp_switch_buf, loop=True)

        # Master volume
        self.master_slider_db = sf.Constant(self.master_volume_slider.value)
        self.master_slider_a = sf.DecibelsToAmplitude(self.master_slider_db)
        self.master_volume = sf.Smooth(self.master_slider_a * self.dsp_switch, samps2mix(24000))

        # Main bus
        self.bus = sf.Bus(num_channels=2)
        self.audio_out = self.bus * self.master_volume

        # Check if HW has 2 channels
        if self.graph.num_output_channels < 2:
            self.audio_out = sf.ChannelMixer(1, self.audio_out)
        self.audio_out.play()


    def enable_dsp(self, state: bool):
        self.dsp_switch_buf.data[0][0] = 1 if state else 0

    
    def add_synth(self, synth):
        self.synths.append(synth)
        self.bus.add_input(synth.output)
    
    def add_feature(self, feature):
        self.features.append(feature)
    
    def add_mapper(self, mapper):
        self.mappers.append(mapper)

    def remove_mapper(self, mapper):
        if mapper in self.mappers:
            self.mappers.remove(mapper)
    
    def compute_features(self, probe_mat):
        for feature in self.features:
            feature(probe_mat)
        
    def compute_mappers(self):
        for mapper in self.mappers:
            mapper()
        

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize(self.image_size)
        self.bg_np = np.array(img, dtype=np.float32) / 255

        # Put the image to the canvas
        img_data = (self.bg_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        img_data = np.transpose(img_data, (1, 0, 2))  # Transpose to match the canvas shape
        self.canvas[0].put_image_data(img_data, self.padding, self.padding)


    def get_probe_matrix(self):
        """Get the probe matrix from the background image."""
        x, y = self.mouse_state[0], self.mouse_state[1]
        probe_w, probe_h = self.probe_state[0], self.probe_state[1]
        x_from = max(x - probe_w//2, 0)
        y_from = max(y - probe_h//2, 0)
        probe = self.bg_np[x_from : x_from + probe_w, y_from : y_from + probe_h]
        return probe
    

    def draw(self):
        """Render new frames for all kernels, then update the HTML canvas with the results."""

        # Get probe matrix
        probe_mat = self.get_probe_matrix()

        # Compute probe features
        self.compute_features(probe_mat)

        # Update mappings
        self.compute_mappers()

        # Clear the canvas
        self.canvas[1].clear()

        # Put the probe rectangle to the canvas
        probe_w, probe_h = self.probe_state[0], self.probe_state[1]
        probe_x, probe_y = self.mouse_state[0], self.mouse_state[1]
        self.canvas[1].stroke_style = 'red' if self.mouse_state[2] > 0 else 'white'
        self.canvas[1].stroke_rect(
            int(probe_x - probe_w//2 + self.padding), 
            int(probe_y - probe_h//2 + self.padding), 
            int(probe_w), 
            int(probe_h))

        # Put the probe to the canvas
        probe_data = (probe_mat * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        probe_data = np.transpose(probe_data, (1, 0, 2))  # Transpose to match the canvas shape
        self.canvas[1].put_image_data(probe_data, self.image_size[0]+20+self.padding, self.padding)


    def mouse_callback(self, x, y, pressed: int = 0):
        """Handle mouse, compute probe features, update synth(s), and render kernels."""

        # Drop excess events over the refresh interval
        current_time = time.time()
        if current_time - self.last_draw_time < self.refresh_interval and pressed < 2: # only skip if mouse is up
            return  # Skip if we are processing too quickly
        self.last_draw_time = current_time  # Update the last event time

        with hold_canvas(self.canvas):

            probe_w, probe_h = self.probe_state[0], self.probe_state[1]
            # clamp x and y to the image size (undo padding) and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
            x_clamped = np.clip(x-self.padding, probe_w//2, self.image_size[0]-1-probe_w//2)
            y_clamped = np.clip(y-self.padding, probe_h//2, self.image_size[1]-1-probe_h//2)
            self.mouse_state[0], self.mouse_state[1] = [x_clamped, y_clamped]
            self.mouse_x_text.value, self.mouse_y_text.value = str(x_clamped), str(y_clamped)

            # Update mouse state
            if pressed == 2:
                self.mouse_state[2] = 1
                self.enable_dsp(True)
            elif pressed == 3:
                self.mouse_state[2] = 0
                self.enable_dsp(False)

            self.draw()


    # GUI callbacks

    # Update probe size from sliders
    def update_probe_width(self, change):
        self.probe_state[0] = change['new']
        self.draw()

    def update_probe_height(self, change):
        self.probe_state[1] = change['new']
        self.draw()

    # Toggle DSP
    def toggle_audio(self, change):
        if change['new']:
            self.graph.start()
            self.audio_graph_toggle.style.text_color = 'green'
        else:
            self.graph.stop()
            self.audio_graph_toggle.style.text_color = 'black'

    def update_master_volume(self, change):
        self.master_slider_db.set_value(change['new'])


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



class Mapper():
    """Map between two buffers. Typically from a feature buffer to a parameter buffer."""
    def __init__(
            self, 
            obj_in, 
            obj_out,
            in_low = None,
            in_high = None,
            out_low = None,
            out_high = None,
            exponent = 1,
            clamp: bool = True,

    ):
        self.obj_in = obj_in
        self.obj_out = obj_out

        # if the input object is an instance of a feature, then we want to map the output of the feature
        # to the input of the object
        if isinstance(self.obj_in, Feature):
            self.buf_in = self.obj_in.features
        elif isinstance(self.obj_in, dict):
            self.buf_in = self.obj_in["buffer"]
        else:
            raise ValueError("Input object must be a Feature or a dict")

        # expecting a synth's param dict here
        self.buf_out = self.obj_out["buffer"]

        # save scaling parameters
        self._in_low = in_low
        self._in_high = in_high
        self._out_low = out_low
        self._out_high = out_high
        self.exponent = exponent
        self.clamp = clamp

        self.create_gui()

    def create_gui(self):
        self.description_text = f"Mapper: {self.obj_in.name} ==> {self.obj_out['name']}"
        self.description_label = widgets.Label(value=self.description_text)
        display(self.description_label)

    def __repr__(self):
        return self.description_text

    @property
    def in_low(self):
        if self._in_low is None:
            if isinstance(self.obj_in, Feature):
                return self.obj_in.min
            elif isinstance(self.obj_in, dict):
                return self.obj_in["min"]
        else:
            return self._in_low
    
    @property
    def in_high(self):
        if self._in_high is None:
            if isinstance(self.obj_in, Feature):
                return self.obj_in.max
            elif isinstance(self.obj_in, dict):
                return self.obj_in["max"]
        else:
            return self._in_high

    @property
    def out_low(self):
        if self._out_low is None:
            return self.obj_out["min"]
        else:
            return self._out_low

    @property
    def out_high(self):
        if self._out_high is None:
            return self.obj_out["max"]
        else:
            return self._out_high

    def map(self):
        # scale the input buffer to the output buffer
        self.buf_out.data[:,:] = scale_array_exp(
            self.buf_in.data,
            self.in_low,
            self.in_high,
            self.out_low,
            self.out_high,
            self.exponent
        )
        if self.clamp:
            self.buf_out.data[:, :] = np.clip(self.buf_out.data[:, :], self.out_low, self.out_high)

    def __call__(self):
        self.map()
