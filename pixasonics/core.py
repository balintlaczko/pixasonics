from .features import Feature
from .utils import scale_array_exp
from .ui import MapperCard, AppUI, ProbeSettings, AudioSettings, Model, find_widget_by_tag
from .utils import samps2mix
from ipycanvas import hold_canvas, MultiCanvas
from IPython.display import display
import time
import numpy as np
import signalflow as sf
from PIL import Image



class App():
    def __init__(
            self,
            image_size: tuple[int] = (500, 500),
            fps: int = 120,
            ):
        
        self.image_size = image_size
        self.refresh_interval = 1 / fps

        # Global state variables
        self.is_drawing = False
        self.last_draw_time = time.time()
        self.bg_np = np.zeros(image_size + (3,), dtype=np.float32) # np.array(img, dtype=np.float32) / 255

        # Private properties
        self._probe_x = 0
        self._probe_y = 0
        self._mouse_btn = 0
        self._probe_width = Model(50)
        self._probe_height = Model(50)
        self._master_volume = Model(0)
        self._audio = Model(0)

        # Containers for features, mappers, and synths
        self.features = []
        self.mappers = []
        self.synths = []

        self.create_ui()
        self.create_audio_graph()

    @property
    def probe_x(self):
        return self._probe_x
    
    @probe_x.setter
    def probe_x(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        x_clamped = np.clip(value, self.probe_width//2, self.image_size[0]-1-self.probe_width//2)
        self._probe_x = int(round(x_clamped))
        self.draw()
    
    @property
    def probe_y(self):
        return self._probe_y
    
    @probe_y.setter
    def probe_y(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        y_clamped = np.clip(value, self.probe_height//2, self.image_size[1]-1-self.probe_height//2)
        self._probe_y = int(round(y_clamped))
        self.draw()

    def update_probe_xy(self):
        self.probe_x = self.probe_x
        self.probe_y = self.probe_y
        # self.draw()

    @property
    def mouse_btn(self):
        return self._mouse_btn
    
    @mouse_btn.setter
    def mouse_btn(self, value):
        self._mouse_btn = value
        self.enable_dsp(True) if value > 0 else self.enable_dsp(False)

    @property
    def probe_width(self):
        return self._probe_width.value
    
    @probe_width.setter
    def probe_width(self, value):
        self._probe_width.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def probe_height(self):
        return self._probe_height.value
    
    @probe_height.setter
    def probe_height(self, value):
        self._probe_height.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def master_volume(self):
        return self._master_volume.value
    
    @master_volume.setter
    def master_volume(self, value):
        self._master_volume.value = value

    @property
    def audio(self):
        return self._audio.value
    
    @audio.setter
    def audio(self, value):
        self._audio.value = value


    def create_ui(self):
        probe_settings = ProbeSettings()
        audio_settings = AudioSettings()
        self.ui = AppUI(probe_settings, audio_settings)()
        display(self.ui)

        # Create the canvas
        self.canvas = MultiCanvas(
            2,
            width=self.image_size[0], 
            height=self.image_size[1])
        app_canvas = find_widget_by_tag(self.ui, "app_canvas")
        app_canvas.children = [self.canvas]

        # Canvas mousing event listeners
        self.canvas.on_mouse_move(lambda x, y: self.mouse_callback(x, y, -1))  # Triggered during mouse movement (keeps track of mouse button state)
        self.canvas.on_mouse_down(lambda x, y: self.mouse_callback(x, y, pressed=2))  # When mouse button pressed
        self.canvas.on_mouse_up(lambda x, y: self.mouse_callback(x, y, pressed=3))  # When mouse button released

        # Bind the probe width and height to the sliders
        probe_w_slider = find_widget_by_tag(self.ui, "probe_w_slider")
        self._probe_width.bind_widget(probe_w_slider, extra_callback=self.update_probe_xy)
        probe_h_slider = find_widget_by_tag(self.ui, "probe_h_slider")
        self._probe_height.bind_widget(probe_h_slider, extra_callback=self.update_probe_xy)

        # Bind the audio toggle and master volume to the widgets
        audio_switch = find_widget_by_tag(self.ui, "audio_switch")
        self._audio.bind_widget(audio_switch, extra_callback=self.toggle_dsp)
        master_volume_slider = find_widget_by_tag(self.ui, "master_volume_slider")
        self._master_volume.bind_widget(master_volume_slider, extra_callback=self.set_master_volume)


    def __call__(self):
        return self.ui
    

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
        self.master_slider_db = sf.Constant(0)
        self.master_slider_a = sf.DecibelsToAmplitude(self.master_slider_db)
        self.master_volume_smooth = sf.Smooth(self.master_slider_a * self.dsp_switch, samps2mix(24000))

        # Main bus
        self.bus = sf.Bus(num_channels=2)
        self.audio_out = self.bus * self.master_volume_smooth

        # Check if HW has 2 channels
        if self.graph.num_output_channels < 2:
            self.audio_out = sf.ChannelMixer(1, self.audio_out)
        self.audio_out.play()


    def enable_dsp(self, state: bool):
        self.dsp_switch_buf.data[0][0] = 1 if state else 0

    
    def attach_synth(self, synth):
        print(f"Attaching {synth}")
        if synth not in self.synths:
            self.synths.append(synth)
            self.bus.add_input(synth.output)
            synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
            synths_carousel.children = list(synths_carousel.children) + [synth.ui]
            synth._ui.app = self

    def detach_synth(self, synth):
        print(f"Detaching {synth}")
        if synth in self.synths:
            self.synths.remove(synth)
            self.bus.remove_input(synth.output)
            synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
            synths_carousel.children = [child for child in synths_carousel.children if child.tag != f"synth_{synth.id}"]
            synth._ui.app = None
    
    def attach_feature(self, feature):
        print(f"Attaching {feature}")
        if feature not in self.features:
            self.features.append(feature)
            features_carousel = find_widget_by_tag(self.ui, "features_carousel")
            features_carousel.children = list(features_carousel.children) + [feature.ui]
            feature._ui.app = self

    def detach_feature(self, feature):
        print(f"Detaching {feature}")
        if feature in self.features:
            self.features.remove(feature)
            features_carousel = find_widget_by_tag(self.ui, "features_carousel")
            features_carousel.children = [child for child in features_carousel.children if child.tag != f"feature_{feature.id}"]
            feature._ui.app = None
    
    def attach_mapper(self, mapper):
        print(f"Attaching {mapper}")
        if mapper not in self.mappers:
            self.mappers.append(mapper)
            mappers_carousel = find_widget_by_tag(self.ui, "mappers_carousel")
            mappers_carousel.children = list(mappers_carousel.children) + [mapper.ui]
            mapper._ui.app = self

    def detach_mapper(self, mapper):
        print(f"Detaching {mapper}")
        if mapper in self.mappers:
            self.mappers.remove(mapper)
            mappers_carousel = find_widget_by_tag(self.ui, "mappers_carousel")
            mappers_carousel.children = [child for child in mappers_carousel.children if child.tag != f"mapper_{mapper.id}"]
            mapper._ui.app = None
    
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
        self.canvas[0].put_image_data(img_data, 0, 0)


    def get_probe_matrix(self):
        """Get the probe matrix from the background image."""
        x_from = max(self.probe_x - self.probe_width//2, 0)
        y_from = max(self.probe_y - self.probe_height//2, 0)
        probe = self.bg_np[y_from : y_from + self.probe_height, x_from : x_from + self.probe_width]
        return probe
    

    def draw(self):
        """Render new frames for all kernels, then update the HTML canvas with the results."""

        # Get probe matrix
        probe_mat = self.get_probe_matrix()

        # Compute probe features
        self.compute_features(probe_mat)

        # Update mappings
        if self.mouse_btn > 0:
            self.compute_mappers()

        # Clear the canvas
        self.canvas[1].clear()

        # Put the probe rectangle to the canvas
        self.canvas[1].stroke_style = 'red' if self.mouse_btn > 0 else 'yellow'
        self.canvas[1].stroke_rect(
            int(self.probe_x - self.probe_width//2), 
            int(self.probe_y - self.probe_height//2), 
            int(self.probe_width), 
            int(self.probe_height))

        # # Put the probe to the canvas
        # probe_data = (probe_mat * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        # self.canvas[1].put_image_data(probe_data, 0, 0)
        # if self.mouse_btn > 0:
        #     display(Image.fromarray(probe_data))


    def mouse_callback(self, x, y, pressed: int = 0):
        """Handle mouse, compute probe features, update synth(s), and render kernels."""

        # Drop excess events over the refresh interval
        current_time = time.time()
        if current_time - self.last_draw_time < self.refresh_interval and pressed < 2: # only skip if mouse is up
            return  # Skip if we are processing too quickly
        self.last_draw_time = current_time  # Update the last event time

        with hold_canvas(self.canvas):
            # Update mouse state
            self.probe_x, self.probe_y = x, y
            if pressed == 2:
                self.mouse_btn = 1
            elif pressed == 3:
                self.mouse_btn = 0
            # Update probe features, mappers, and render canvas
            self.draw()


    # GUI callbacks

    def toggle_dsp(self):
        if self.audio > 0:
            self.graph.start()
            audio_switch = find_widget_by_tag(self.ui, "audio_switch")
            audio_switch.style.text_color = 'green'
        else:
            self.graph.stop()
            audio_switch = find_widget_by_tag(self.ui, "audio_switch")
            audio_switch.style.text_color = 'black'

    def set_master_volume(self):
        self.master_slider_db.set_value(self.master_volume)


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
        # self.buf_out = self.obj_out["buffer"]
        self.obj_out_owner = self.obj_out["owner"]

        # save scaling parameters
        self._in_low = in_low
        self._in_high = in_high
        self._out_low = out_low
        self._out_high = out_high
        self.exponent = exponent
        self.clamp = clamp

        self.id = str(id(self))

        self._ui = MapperCard(
            id=self.id,
            from_name=self.obj_in.name,
            to_name=self.obj_out["name"],
        )
        self._ui.mapper = self

    @property
    def ui(self):
        return self._ui()

    def __repr__(self):
        return f"Mapper {self.id}: {self.obj_in.name} -> {self.obj_out['name']}"

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

    # def map(self):
    #     # scale the input buffer to the output buffer
    #     self.buf_out.data[:,:] = scale_array_exp(
    #         self.buf_in.data,
    #         self.in_low,
    #         self.in_high,
    #         self.out_low,
    #         self.out_high,
    #         self.exponent
    #     )
    #     if self.clamp:
    #         self.buf_out.data[:, :] = np.clip(self.buf_out.data[:, :], self.out_low, self.out_high)

    def map(self):
        # scale the input buffer to the output buffer
        scaled_val = scale_array_exp(
            self.buf_in.data,
            self.in_low,
            self.in_high,
            self.out_low,
            self.out_high,
            self.exponent
        )
        if self.clamp:
            scaled_val = np.clip(scaled_val, self.out_low, self.out_high)

        self.obj_out_owner.set_input_buf(
            self.obj_out["param_name"],
            scaled_val,
            from_slider=False
        )

    def __call__(self):
        self.map()
