from .features import Feature
from .utils import scale_array_exp, sec2frame, resize_interp, samps2mix
from .ui import MapperCard, AppUI, ImageSettings, ProbeSettings, AudioSettings, Model, find_widget_by_tag
from .synths import Synth, Envelope
from ipycanvas import hold_canvas, MultiCanvas
from IPython.display import display
import time
import numpy as np
import signalflow as sf
from PIL import Image
import threading
from typing import List, Tuple, Dict, Optional, Union

class AppRegistry:
    _instance = None
    _apps = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, app):
        self._apps.add(app)

    def unregister(self, app):
        self._apps.discard(app)

    def notify_reregister(self, notifier):
        for app in self._apps:
            if app != notifier:
                app.create_audio_graph()

    def notify_pause(self, notifier):
        for app in self._apps:
            if app != notifier:
                app._audio_prev = app.audio
                app.audio = False

    def notify_resume(self, notifier):
        for app in self._apps:
            if app != notifier:
                app.audio = app._audio_prev
                app._audio_prev = None

class App():
    def __init__(
            self,
            image_size: tuple[int] = (500, 500),
            fps: int = 60,
            nrt: bool = False,
            # output_buffer_size: int = 480,
            headless: bool = False,
            ):
        
        self.image_size = image_size

        # threading
        self.compute_thread = None
        self.compute_lock = threading.Lock()
        self.compute_event = threading.Event()
        self.stop_event = threading.Event()

        # Global state variables
        self.is_drawing = False
        self.last_draw_time = time.time()
        self.bg_hires = np.zeros(image_size + (3,), dtype=np.float64)
        self.bg_display = np.zeros(image_size + (3,), dtype=np.uint8)

        # Private properties
        self._fps = fps
        self._refresh_interval = 1 / fps
        self._probe_x = 0
        self._probe_x_on_last_draw = 0
        self._probe_y = 0
        self._probe_y_on_last_draw = 0
        self._mouse_btn = 0
        self._probe_width = Model(50)
        self._probe_width_on_last_draw = 50
        self._probe_height = Model(50)
        self._probe_height_on_last_draw = 50
        self._probe_follows_idle_mouse = Model(False)
        self._interaction_mode = Model("Hold")
        self._last_mouse_down_time = 0
        self._master_volume = Model(0)
        self._audio = Model(False)
        self._recording = Model(False)
        self._recording_path = Model("recording.wav")
        self._unmuted = False
        self._unmuted_on_last_draw = False
        self._nrt = nrt
        self._output_buffer_size = 480 # output_buffer_size
        self._sample_rate = 48000 # sample_rate
        self._normalize_display = Model(False)
        self._normalize_display_global = Model(False)
        self._display_channel_offset = Model(0)
        self._display_layer_offset = Model(0)
        self._image_is_loaded = False
        self._headless = headless
        self._draw_lock = False # True disables self.draw()

        # Containers for features, mappers, and synths
        self.features = []
        self.mappers = []
        self.synths = []

        self.ui = None
        if not self._headless:
            self.create_ui()
        self.create_audio_graph()
        self.start_compute_thread()

        AppRegistry().register(self)

    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._fps = value
        self._refresh_interval = 1 / value

    @property
    def normalize_display(self):
        return self._normalize_display.value
    
    @normalize_display.setter
    def normalize_display(self, value):
        self._normalize_display.value = value
        self.redraw_background()

    @property
    def normalize_display_global(self):
        return self._normalize_display_global.value
    
    @normalize_display_global.setter
    def normalize_display_global(self, value):
        self._normalize_display_global.value = value
        self.redraw_background()

    @property
    def display_channel_offset(self):
        return self._display_channel_offset.value
    
    @display_channel_offset.setter
    def display_channel_offset(self, value):
        self._display_channel_offset.value = value
        self.redraw_background()

    @property
    def display_layer_offset(self):
        return self._display_layer_offset.value
    
    @display_layer_offset.setter
    def display_layer_offset(self, value):
        self._display_layer_offset.value = value
        self.redraw_background()

    @property
    def image(self):
        return self.bg_hires
    
    @property
    def image_displayed(self):
        return self.bg_display

    @property
    def nrt(self):
        return self._nrt
    
    @nrt.setter
    def nrt(self, value):
        changed = value != self._nrt
        self._nrt = value
        if changed:
            if value:
                # when swtiched to NRT, remove our graph from the global graph
                self.audio = False
        # enable/disable the Audio button
        self.toggle_audio_btn(not value)
        # notify mappers
        for mapper in self.mappers:
            mapper.nrt = value

    @property
    def probe_follows_idle_mouse(self):
        return self._probe_follows_idle_mouse.value
    
    @probe_follows_idle_mouse.setter
    def probe_follows_idle_mouse(self, value):
        self._probe_follows_idle_mouse.value = value

    def clamp_probe_x(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        x_clamped = np.clip(value, self.probe_width//2, self.image_size[1]-1-self.probe_width//2)
        return int(round(x_clamped))
    
    def clamp_probe_y(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        y_clamped = np.clip(value, self.probe_height//2, self.image_size[0]-1-self.probe_height//2)
        return int(round(y_clamped))

    @property
    def probe_x(self):
        return self._probe_x
    
    @probe_x.setter
    def probe_x(self, value):
        self._probe_x = self.clamp_probe_x(value)
        if not self._nrt:
            self.draw()
    
    @property
    def probe_y(self):
        return self._probe_y
    
    @probe_y.setter
    def probe_y(self, value):
        self._probe_y = self.clamp_probe_y(value)
        if not self._nrt:
            self.draw()

    def update_probe_xy(self):
        # Apply the clamped probe position without triggering a draw
        self._probe_x = self.clamp_probe_x(self.probe_x)
        self._probe_y = self.clamp_probe_y(self.probe_y)
        if not self._nrt:
            self.draw()

    @property
    def mouse_btn(self):
        return self._mouse_btn
    
    @mouse_btn.setter
    def mouse_btn(self, value):
        self._mouse_btn = value
        if self.interaction_mode == "Hold":
            self.unmuted = value > 0
        elif self.interaction_mode == "Toggle" and value > 1: # double-click
            self.unmuted = not self.unmuted

    @property
    def probe_width(self):
        return int(self._probe_width.value)
    
    @probe_width.setter
    def probe_width(self, value):
        self._probe_width.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def probe_height(self):
        return int(self._probe_height.value)
    
    @probe_height.setter
    def probe_height(self, value):
        self._probe_height.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def _probe_changed(self):
        return (
            self._probe_x != self._probe_x_on_last_draw
            or self._probe_y != self._probe_y_on_last_draw
            or self._probe_width != self._probe_width_on_last_draw
            or self._probe_height != self._probe_height_on_last_draw
        )

    @property
    def master_volume(self):
        return self._master_volume.value
    
    @master_volume.setter
    def master_volume(self, value):
        self._master_volume.value = value
        self.set_master_volume()

    @property
    def audio(self):
        return self._audio.value
    
    @audio.setter
    def audio(self, value):
        self._audio.value = value
        self.toggle_dsp()

    @property
    def recording(self):
        return self._recording.value
    
    @recording.setter
    def recording(self, value):
        self._recording.value = value
        self.toggle_record()

    @property
    def recording_path(self):
        return self._recording_path.value
    
    @recording_path.setter
    def recording_path(self, value):
        if not value.endswith(".wav"):
            value = value + ".wav"
        # only update if the value is different
        if value != self._recording_path.value:
            self._recording_path.value = value

    @property
    def unmuted(self):
        return self._unmuted
    
    @unmuted.setter
    def unmuted(self, value):
        self._unmuted = value
        if value:
            self.master_envelope.on()
        else:
            self.master_envelope.off()

    @property
    def _unmuted_changed(self):
        return self._unmuted != self._unmuted_on_last_draw
    
    @property
    def output_buffer_size(self):
        if self.graph is not None:
            return self.graph.output_buffer_size
        else:
            return None
        
    # @output_buffer_size.setter
    # def output_buffer_size(self, value):
    #     print(f"Setting output buffer size to {value}")
    #     self._output_buffer_size = value
    #     print(f"Destroying audio graph")
    #     self.graph.destroy()
    #     print(f"Creating new audio graph")
    #     self.create_audio_graph()
    #     # print(f"Re-registering app")
    #     # AppRegistry().notify_reregister(self)

    @property
    def sample_rate(self):
        if self.graph is not None:
            return self.graph.sample_rate
        else:
            return None

    @property
    def interaction_mode(self):
        return self._interaction_mode.value
    
    @interaction_mode.setter
    def interaction_mode(self, value):
        self._interaction_mode.value = value.capitalize()


    def start_compute_thread(self):
        if self.compute_thread is None or not self.compute_thread.is_alive():
            self.stop_event.clear()
            self.compute_thread = threading.Thread(target=self.compute_loop, daemon=True)
            self.compute_thread.start()

    def stop_compute_thread(self):
        self.stop_event.set()
        self.compute_event.set()
        if self.compute_thread is not None:
            self.compute_thread.join()

    def compute_loop(self):
        while not self.stop_event.is_set():
            self.compute_event.wait()
            self.compute_event.clear()
            with self.compute_lock:
                probe_mat = self.get_probe_matrix()
                self.compute_features(probe_mat)
                if self.unmuted:
                    self.compute_mappers()
            time.sleep(0.01)  # Small sleep to prevent busy-waiting

    def cleanup(self):
        self.stop_compute_thread()
        try:
            self.audio_out.stop()
        except sf.NodeNotPlayingException:
            pass
        AppRegistry().unregister(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def __del__(self):
        self.cleanup()
    

    def create_ui(self):
        image_settings = ImageSettings()
        probe_settings = ProbeSettings(
            canvas_width=self.image_size[1],
            canvas_height=self.image_size[0]
        )
        audio_settings = AudioSettings()
        self.ui = AppUI(
            audio_settings, 
            image_settings, 
            probe_settings,
            canvas_height=self.image_size[0],
            canvas_width=self.image_size[1])()
        display(self.ui)

        # Create the canvas
        self.canvas = MultiCanvas(
            2,
            width=self.image_size[1], 
            height=self.image_size[0])
        app_canvas = find_widget_by_tag(self.ui, "app_canvas")
        app_canvas.children = [self.canvas]

        # Canvas mousing event listeners
        self.canvas.on_mouse_move(lambda x, y: self.mouse_callback(x, y, -1))  # Triggered during mouse movement (keeps track of mouse button state)
        self.canvas.on_mouse_down(lambda x, y: self.mouse_callback(x, y, pressed=2))  # When mouse button pressed
        self.canvas.on_mouse_up(lambda x, y: self.mouse_callback(x, y, pressed=3))  # When mouse button released

        # Bind image settings widgets
        chkbox_normalize_display = find_widget_by_tag(self.ui, "normalize_display")
        self._normalize_display.bind_widget(chkbox_normalize_display, extra_callback=self.redraw_background)
        chkbox_normalize_display_global = find_widget_by_tag(self.ui, "normalize_display_global")
        self._normalize_display_global.bind_widget(chkbox_normalize_display_global, extra_callback=self.redraw_background)
        channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
        self._display_channel_offset.bind_widget(channel_offset_slider, extra_callback=self.redraw_background)
        layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
        self._display_layer_offset.bind_widget(layer_offset_slider, extra_callback=self.redraw_background)

        # Bind the probe settings widgets
        # Probe sliders
        probe_w_slider = find_widget_by_tag(self.ui, "probe_w_slider")
        self._probe_width.bind_widget(probe_w_slider, extra_callback=self.update_probe_xy)
        probe_h_slider = find_widget_by_tag(self.ui, "probe_h_slider")
        self._probe_height.bind_widget(probe_h_slider, extra_callback=self.update_probe_xy)
        # Interaction mode buttons
        interaction_mode_buttons = find_widget_by_tag(self.ui, "interaction_mode_buttons")
        self._interaction_mode.bind_widget(interaction_mode_buttons)
        # Follow idle mouse checkbox
        chkbox_probe_follows_idle_mouse = find_widget_by_tag(self.ui, "probe_follows_idle_mouse")
        self._probe_follows_idle_mouse.bind_widget(chkbox_probe_follows_idle_mouse)

        # Bind the audio settings widgets
        # Audio switch and master volume slider
        audio_switch = find_widget_by_tag(self.ui, "audio_switch")
        self._audio.bind_widget(audio_switch, extra_callback=self.toggle_dsp)
        master_volume_slider = find_widget_by_tag(self.ui, "master_volume_slider")
        self._master_volume.bind_widget(master_volume_slider, extra_callback=self.set_master_volume)
        # Recording toggle and file path
        recording_toggle = find_widget_by_tag(self.ui, "recording_toggle")
        self._recording.bind_widget(recording_toggle, extra_callback=self.toggle_record)
        recording_path = find_widget_by_tag(self.ui, "recording_path")
        self._recording_path.bind_widget(recording_path)



    def __call__(self):
        return self.ui
    

    def create_audio_graph(self):
        # Get or create the shared audio graph
        self.graph = sf.AudioGraph.get_shared_graph()
        if self.graph is None:
            config = sf.AudioGraphConfig()
            config.output_buffer_size = self._output_buffer_size
            config.sample_rate = self._sample_rate
            self.graph = sf.AudioGraph(config=config, start=True)


        # Master volume
        self.master_slider_db = sf.Constant(0)
        self.master_slider_a = sf.DecibelsToAmplitude(self.master_slider_db)
        self.master_volume_smooth = sf.Smooth(self.master_slider_a, samps2mix(24000))

        # Master envelope
        self.master_envelope = Envelope(
            attack=0.1,
            decay=0.01,
            sustain=1,
            release=0.1,
            name="Master Envelope"
        )
        self.master_envelope_bus = sf.Bus(1)
        self.master_envelope_bus.add_input(self.master_envelope.output)
        if not self._headless:
            # add ui to the app ui
            audio_settings = find_widget_by_tag(self.ui, "audio_settings")
            # always keep the first 2 children only, and replace the rest with this env ui
            audio_settings.children = [*audio_settings.children[:2], self.master_envelope.ui]

        # Main bus
        self.bus = sf.Bus(num_channels=2)
        self.audio_out = self.bus * self.master_volume_smooth * self.master_envelope_bus

        # if HW has 2 channels downmix to mono
        if self.graph.num_output_channels < 2:
            self.audio_out = sf.ChannelMixer(1, self.audio_out)

        # Add any registered synths to the bus
        # TODO: with new synths (based on specs) we'll have to re-instantiate them here on a potentially new graph
        for synth in self.synths:
            self.bus.add_input(synth.output)

        # put the app's graph to the global graph if audio is enabled
        # but not in nrt, because in that case we don't want to play the audio
        if self.audio > 0 and not self.nrt:
            self.audio_out.play()
            self.unmuted = self.unmuted # call the setter to update the envelope state

    
    def attach_synth(self, synth):
        #print(f"Attaching {synth}")
        if synth not in self.synths:
            self.synths.append(synth)
            self.bus.add_input(synth.output)
            if not self._headless:
                synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
                synths_carousel.children = list(synths_carousel.children) + [synth.ui]
                synth._ui.app = self

    def detach_synth(self, synth):
        #print(f"Detaching {synth}")
        if synth in self.synths:
            self.synths.remove(synth)
            self.bus.remove_input(synth.output)
            if not self._headless:
                synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
                synths_carousel.children = [child for child in synths_carousel.children if child.tag != f"synth_{synth.id}"]
                synth._ui.app = None
    
    def attach_feature(self, feature):
        #print(f"Attaching {feature}")
        if feature not in self.features:
            self.features.append(feature)
            feature.app = self
            if not self._headless:
                features_carousel = find_widget_by_tag(self.ui, "features_carousel")
                features_carousel.children = list(features_carousel.children) + [feature.ui]
                feature._ui.app = self

    def detach_feature(self, feature):
        #print(f"Detaching {feature}")
        if feature in self.features:
            self.features.remove(feature)
            if not self._headless:
                features_carousel = find_widget_by_tag(self.ui, "features_carousel")
                features_carousel.children = [child for child in features_carousel.children if child.tag != f"feature_{feature.id}"]
                feature._ui.app = None
    
    def attach_mapper(self, mapper):
        #print(f"Attaching {mapper}")
        if mapper not in self.mappers:
            self.mappers.append(mapper)
            mapper._app = self
            if not self._headless:
                mappers_carousel = find_widget_by_tag(self.ui, "mappers_carousel")
                mappers_carousel.children = list(mappers_carousel.children) + [mapper.ui]
                mapper._ui.app = self
            # evaluate once to trigger JIT compilation
            mapper()

    def detach_mapper(self, mapper):
        #print(f"Detaching {mapper}")
        if mapper in self.mappers:
            self.mappers.remove(mapper)
            mapper._app = None
            if not self._headless:
                mappers_carousel = find_widget_by_tag(self.ui, "mappers_carousel")
                mappers_carousel.children = [child for child in mappers_carousel.children if child.tag != f"mapper_{mapper.id}"]
                mapper._ui.app = None

    def attach(self, obj):
        if isinstance(obj, Feature):
            self.attach_feature(obj)
        elif isinstance(obj, Mapper):
            self.attach_mapper(obj)
        elif isinstance(obj, Synth):
            self.attach_synth(obj)
        else:
            raise ValueError(f"Cannot attach object of type {type(obj)}")
        
    def detach(self, obj):
        if isinstance(obj, Feature):
            self.detach_feature(obj)
        elif isinstance(obj, Mapper):
            self.detach_mapper(obj)
        elif isinstance(obj, Synth):
            self.detach_synth(obj)
        else:
            raise ValueError(f"Cannot detach object of type {type(obj)}")
    
    def compute_features(self, probe_mat):
        for feature in self.features:
            feature(probe_mat)
        
    def compute_mappers(self, frame=None):
        for mapper in self.mappers:
            mapper(frame)
        

    def load_image_file(self, image_path: str, refresh_features: bool = True) -> None:
        img = Image.open(image_path)
        if img.size != self.image_size:
            img = img.resize(self.image_size[::-1]) # PIL uses (W, H) instead of (H, W)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[..., None, None] # add channel and layer dimensions if single-channel
        elif len(img.shape) == 3:
            img = img[..., None] # add layer dimension if 3-channel
        # print ("Image shape:", img.shape)
        self.bg_hires = img
        self.bg_display = None
        if not self._headless:
            self.bg_display = self.convert_image_data_for_display(
                self.bg_hires, 
                normalize=self.normalize_display, 
                global_normalize=self.normalize_display_global)
        
        self._image_is_loaded = True

        if not self._headless:
            # Set layer offset to 0 and disable the slider
            self._display_layer_offset.value = 0
            layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
            layer_offset_slider.disabled = True
            layer_offset_slider.max = 0

            # Set the channel offset to 0 and disable the slider
            self._display_channel_offset.value = 0
            channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
            channel_offset_slider.disabled = True
            channel_offset_slider.max = 0

            # Redraw the background with the new image
            self.redraw_background()

        # re-trigger image processing in already attached features
        if refresh_features:
            for feature in self.features:
                feature.app = self


    def load_image_data(self, img_data: np.ndarray, refresh_features: bool = True) -> None:
        if img_data.shape[0:2] != self.image_size:
            img_data = self.resize_image_data(img_data)
        self.bg_hires = img_data
        self.bg_display = self.convert_image_data_for_display(
            self.bg_hires, 
            normalize=self.normalize_display, 
            global_normalize=self.normalize_display_global)
        
        self._image_is_loaded = True

        if not self._headless:
            # Set layer offset to 0, enable the slider, and set the max value
            if len(self.bg_hires.shape) == 4 and self.bg_hires.shape[3] > 1:
                self._display_layer_offset.value = 0
                layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
                layer_offset_slider.disabled = False
                layer_offset_slider.max = self.bg_hires.shape[-1] - 1
            else:
                self._display_layer_offset.value = 0
                layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
                layer_offset_slider.disabled = True
                layer_offset_slider.max = 0

            # Set the channel offset to 0, enable the slider, and set the max value
            if self.bg_hires.shape[2] > 3:
                self._display_channel_offset.value = 0
                channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
                channel_offset_slider.disabled = False
                channel_offset_slider.max = self.bg_hires.shape[2] - 3
            else:
                self._display_channel_offset.value = 0
                channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
                channel_offset_slider.disabled = True
                channel_offset_slider.max = 0

            self.redraw_background()

        # re-trigger image processing in already attached features
        if refresh_features:
            for feature in self.features:
                feature.app = self

    
    def resize_image_data(self, img_data: np.ndarray) -> np.ndarray:
        # if 3D, add a layer dimension
        if len(img_data.shape) == 3:
            img_data = img_data[..., None]
        # loop through the layers and resize each one
        img_data_resized = np.zeros(self.image_size + img_data.shape[2:], dtype=img_data.dtype)
        for i in range(img_data.shape[3]):
            layer = img_data[:, :, :, i]
            resized_layer = np.zeros(self.image_size + (layer.shape[2],), dtype=img_data.dtype)
            for j in range(layer.shape[2]):
                img = Image.fromarray(layer[:, :, j])
                resized_layer[:, :, j] = np.array(img.resize(self.image_size[::-1])) # PIL uses (W, H) instead of (H, W)
            img_data_resized[:, :, :, i] = resized_layer
        return img_data_resized


    def convert_image_data_for_display(
            self, 
            img_data, 
            normalize=False, 
            global_normalize=False,
            channel_offset=0,
            layer_offset=0
            ):
        img = self.rescale_image_data_for_display(
            img_data, 
            normalize=normalize, 
            global_normalize=global_normalize)
        # if 4D, slice the layer according to the layer offset
        if len(img.shape) == 4:
            img = img[:, :, :, layer_offset]
        # if single channel, repeat to 3 channels
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        # if two channels, add a third empty channel
        elif img.shape[2] == 2:
            img = np.concatenate([img, np.zeros(img.shape[:2] + (1,), dtype=img.dtype)], axis=2)
        # if more than 3 channels, slice 3 channels according to the channel offset
        elif img.shape[2] > 3:
            img = img[:, :, channel_offset:channel_offset+3]
        return img


    def rescale_image_data_for_display(self, img_data, normalize=False, global_normalize=False):
        if normalize:
            if global_normalize:
                return ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
            else:
                # if 3D
                if len(img_data.shape) == 3: # H, W, C
                    return ((img_data - img_data.min(axis=(0, 1))) / (img_data.max(axis=(0, 1)) - img_data.min(axis=(0, 1))) * 255).astype(np.uint8)
                # if 4D
                elif len(img_data.shape) == 4: # H, W, C, L
                    img_min = img_data.min(axis=(0, 1, 3))[..., None] # reduce H, W, L
                    img_max = img_data.max(axis=(0, 1, 3))[..., None]
                    return ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        # if not normalizing then divide by max value of the data type
        if np.issubdtype(img_data.dtype, np.integer):
            return (img_data / np.iinfo(img_data.dtype).max * 255).astype(np.uint8)
        else:
            return (img_data / np.finfo(img_data.dtype).max * 255).astype(np.uint8)
        
    
    def redraw_background(self):
        if self._headless:
            return
        if not self._image_is_loaded:
            return
        self.bg_display = self.convert_image_data_for_display(
            self.bg_hires, 
            normalize=self._normalize_display.value, 
            global_normalize=self._normalize_display_global.value,
            channel_offset=self.display_channel_offset,
            layer_offset=self.display_layer_offset
            )
        self.canvas[0].put_image_data(self.bg_display, 0, 0)


    def get_probe_matrix(self) -> np.ndarray:
        """
        Get the probe matrix from the background image.
        The probe matrix is a square region of the background image.
        The size of the probe is determined by the probe_width and probe_height properties.
        The position of the probe is determined by the probe_x and probe_y properties.
        The probe is clamped to the image size, so that it doesn't go out of bounds.

        Returns:
            np.ndarray: The probe matrix.
        """
        x_from = max(self.probe_x - self.probe_width//2, 0)
        y_from = max(self.probe_y - self.probe_height//2, 0)
        probe = self.bg_hires[y_from : y_from + self.probe_height, x_from : x_from + self.probe_width]
        return probe
    

    def render_timeline_to_array(self, timeline: List[Tuple[float, Dict]]) -> np.ndarray:
        """
        Render the timeline to a new buffer and return it as a numpy array.
        This function is for NRT rendering a timeline, which can be considered
        as an "automation" for the size and position of the Probe. More precisely,
        the timeline is a list of tuples, where each tuple contains the time
        in seconds and the Probe settings for that time as a dictionary.
        Args:
            timeline (List[Tuple[float, Dict]]): The timeline to render. Each tuple contains the time in seconds and the Probe settings for that time.
        Returns:
            np.ndarray: The rendered audio buffer as a numpy array of shape (n_channels, n_samples), where n_channels is always 2 (i.e. stereo).
        """
        out_buf = self.render_timeline(timeline)
        arr = np.copy(out_buf.data)
        return arr
    

    def render_timeline_to_file(
            self, 
            timeline: List[Tuple[float, Dict]],
            target_filename: str,
            ) -> None:
        """
        Render the timeline to a file. 
        This function is for NRT rendering a timeline, which can be considered
        as an "automation" for the size and position of the Probe. More precisely,
        the timeline is a list of tuples, where each tuple contains the time
        in seconds and the Probe settings for that time as a dictionary.
        The timeline is rendered to a new buffer and saved to the target filename.
        The target filename must end with .wav.

        Args:
            timeline (List[Tuple[float, Dict]]): The timeline to render. Each tuple contains the time in seconds and the Probe settings for that time.
            target_filename (str): The target filename to save the rendered audio to. The filename must end with .wav.
        """
        out_buf = self.render_timeline(timeline)
        out_buf.save(target_filename)
    

    def render_timeline(self, timeline: List[Tuple[float, Dict]]) -> sf.Buffer:
        """
        Render the timeline to a new buffer.
        This function is for NRT rendering a timeline, which can be considered
        as an "automation" for the size and position of the Probe. More precisely,
        the timeline is a list of tuples, where each tuple contains the time
        in seconds and the Probe settings for that time as a dictionary.

        Args:
            timeline (List[Tuple[float, Dict]]): The timeline to render. Each tuple contains the time in seconds and the Probe settings for that time.

        Returns:
            sf.Buffer: The rendered audio buffer.
        """
        # create an output buffer to store the rendered audio
        last_time_s, _ = timeline[-1]
        self._output_samps = int(np.ceil(last_time_s * self.graph.sample_rate))
        self._render_nframes = int(np.ceil(last_time_s * self.fps))
        _output_buffer = sf.Buffer(2, self._output_samps)
        _output_buffer.sample_rate = self.graph.sample_rate

        # step 1: save the before state, stop compute thread, enable draw lock, generate timeline frames, pause other apps
        # save the before state
        self._nrt_prev = self._nrt # save current nrt state for later
        self._audio_prev = self.audio # save current audio state for later
        self._unmuted_prev = self.unmuted # save current unmuted state for later
        probe_x_prev = self.probe_x
        probe_y_prev = self.probe_y
        probe_width_prev = self.probe_width
        probe_height_prev = self.probe_height
        # stop compute thread
        self.stop_compute_thread()
        # enable draw lock
        self._draw_lock = True
        # generate automation lines
        timeline_frames = self.generate_timeline_frames(timeline, self._render_nframes, self.fps)
        # stop audio in this app and in other apps
        self.nrt = False # not yet, we need mappers to set the synths directly
        self.audio = False
        AppRegistry().notify_pause(self) # notify other apps to pause their audio
        self.graph.stop()
        self.graph.render_to_new_buffer(1) # "sync" the graph

        # step 2: initialize mappings & synths to the first frame 
        # This is to avoid starting with an interpolation from wherever the Probe was before calling the render.
        # Also to set the master envelope to 0
        first_frame_settings = {key: val[0] for key, val in timeline_frames.items()}
        # apply settings from first frame
        self.render_frame(None, first_frame_settings) # trigger all mappings based on the Probe matrix of the first frame
        self.audio = True # put audio on the graph
        self.graph.render_to_new_buffer(1) # "sync" the graph
        self.master_envelope.set_input("gate", 0) # stop the env
        master_envelope_release_time_samps = int(np.ceil(self.master_envelope.release * self.graph.sample_rate))
        dummy_length_samps = master_envelope_release_time_samps + 1000 # add a bit of padding to the dummy length
        self.graph.render_to_new_buffer(dummy_length_samps)

        # step 3: trigger the envelope and render everything until the start of the release
        # switch on NRT mode
        self.nrt = True # call setter to turn off audio to notify mappers
        self.graph.render_to_new_buffer(1) # "sync" the graph
        # render the timeline (record mappings for each frame)
        for frame in range(self._render_nframes):
            frame_settings = {key: val[frame] for key, val in timeline_frames.items()}
            self.render_frame(frame, frame_settings)
        self.audio = True # put audio on the graph
        self.master_envelope.set_input("gate", 1) # start the env
        render_part1_length_samps = self._output_samps - master_envelope_release_time_samps
        part1_buffer = self.graph.render_to_new_buffer(render_part1_length_samps)

        # step 4: render the release time
        render_part2_length_samps = master_envelope_release_time_samps
        self.master_envelope.set_input("gate", 0) # stop the env
        # render the release time
        part2_buffer = self.graph.render_to_new_buffer(render_part2_length_samps)
        # combine the two buffers
        _output_buffer.data[:, :] = np.concatenate([part1_buffer.data, part2_buffer.data], axis=1)

        # step 5: restore the before state
        self.nrt = self._nrt_prev # call setter to set audio btn and notify mappers
        self.audio = self._audio_prev
        self.unmuted = self._unmuted_prev
        # restore the Probe state
        self.probe_width = probe_width_prev
        self.probe_height = probe_height_prev
        self.probe_x = probe_x_prev
        self.probe_y = probe_y_prev
        AppRegistry().notify_resume(self) # notify other apps to resume their audio
        self.graph.start() # start the global graph
        self._draw_lock = False # disable draw lock
        self.start_compute_thread() # start the compute thread
        self.draw() # trigger a draw to update the canvas and update mappings to where they where before the render

        return _output_buffer


    def render_frame(self, frame, settings):
        # set the app to the settings
        self.probe_x = settings["probe_x"]
        self.probe_y = settings["probe_y"]
        self.probe_width = settings["probe_width"]
        self.probe_height = settings["probe_height"]

        # Get probe matrix
        probe_mat = self.get_probe_matrix()

        # Compute probe features
        self.compute_features(probe_mat)

        # Update mappings
        self.compute_mappers(frame=frame)


    def standardize_timeline(self, timeline):
        """Fill in missing values in the timeline with the previous values."""
        latest_setting = {
            "probe_width": 1,
            "probe_height": 1,
            "probe_x": 0,
            "probe_y": 0,
        }
        new_timeline = []
        for timepoint, settings in timeline:
            new_settings = {**latest_setting, **settings}
            new_timeline.append((timepoint, new_settings))
            latest_setting = new_settings
        return new_timeline


    def generate_timeline_frames(self, timeline, num_frames, fps):
        # initialize the timeline arrays
        timeline_frames = {
            "probe_width": np.zeros(num_frames),
            "probe_height": np.zeros(num_frames),
            "probe_x": np.zeros(num_frames),
            "probe_y": np.zeros(num_frames),
        }

        standardized_timeline = self.standardize_timeline(timeline)

        # fill the timeline arrays
        for i in range(len(timeline) - 1):
            current_time, current_settings = standardized_timeline[i]
            next_time, next_settings = standardized_timeline[i+1]
            current_frame = sec2frame(current_time, fps)
            next_frame = sec2frame(next_time, fps)
            n_frames = next_frame - current_frame

            for key in timeline_frames.keys():
                current_val = current_settings[key]
                next_val = next_settings[key]
                timeline_frames[key][current_frame:next_frame] = np.linspace(current_val, next_val, n_frames)

        return timeline_frames


    def draw(self):
        """Render new frames for all kernels, then update the HTML canvas with the results."""
        if self._draw_lock:
            return
        
        # print("Drawing...")
        # Signal the compute thread to start processing
        self.compute_event.set()

        # Escape in headless mode
        if self._headless:
            return
        
        # Clear the canvas
        self.canvas[1].clear()

        # Put the probe rectangle to the canvas
        self.canvas[1].stroke_style = 'red' if self.unmuted else 'yellow'
        self.canvas[1].stroke_rect(
            int(self.probe_x - self.probe_width//2), 
            int(self.probe_y - self.probe_height//2), 
            int(self.probe_width), 
            int(self.probe_height))
        
        # update the probe_x and probe_y values in the UI
        probe_x_numbox = find_widget_by_tag(self.ui, "probe_x")
        probe_x_numbox.value = self.probe_x
        probe_y_numbox = find_widget_by_tag(self.ui, "probe_y")
        probe_y_numbox.value = self.probe_y

        # log probe params and unmuted state
        self._probe_x_on_last_draw = self._probe_x
        self._probe_y_on_last_draw = self._probe_y
        self._probe_width_on_last_draw = self._probe_width
        self._probe_height_on_last_draw = self._probe_height
        self._unmuted_on_last_draw = self._unmuted


    def mouse_callback(self, x, y, pressed: int = 0):
        """Handle mouse, compute probe features, update synth(s), and render kernels."""
        if self._nrt:
            return # Skip if we are in non-real-time mode
        
        if not self.probe_follows_idle_mouse and pressed < 0 and self.mouse_btn == 0:
            return # Skip if we are not following the idle mouse

        # Drop excess events over the refresh interval
        current_time = time.time()
        if current_time - self.last_draw_time < self._refresh_interval and pressed < 2: # only skip if mouse is up
            return  # Skip if we are processing too quickly
        self.last_draw_time = current_time  # Update the last event time

        with hold_canvas(self.canvas):
            # Update probe position without triggering a draw
            self._probe_x = self.clamp_probe_x(x)
            self._probe_y = self.clamp_probe_y(y)
            if pressed == 2:
                if current_time - self._last_mouse_down_time < 0.2:
                    self.mouse_btn = 2 # Double-click
                else:
                    self.mouse_btn = 1 # Single-click
                self._last_mouse_down_time = current_time
            elif pressed == 3:
                self.mouse_btn = 0
            # Update probe features, mappers, and render canvas
            # only draw when any of the probe params or unmuted has changed since the last draw
            if self._probe_changed or self._unmuted_changed:
                self.draw()


    # GUI callbacks

    def toggle_dsp(self):
        if not self._headless:
            audio_switch = find_widget_by_tag(self.ui, "audio_switch")
        if self.audio:
            try:
                self.audio_out.play()
            except sf.NodeAlreadyPlayingException:
                pass
            if not self._headless:
                audio_switch.style.text_color = 'green'
        else:
            try:
                self.audio_out.stop()
            except sf.NodeNotPlayingException:
                pass
            if not self._headless:
                audio_switch.style.text_color = 'black'

    def toggle_audio_btn(self, value: bool):
        if self._headless:
            return
        audio_switch = find_widget_by_tag(self.ui, "audio_switch")
        if value:
            audio_switch.disabled = False
            audio_switch.description = 'Audio'
        else:
            audio_switch.disabled = True
            audio_switch.description = 'NRT'

    def toggle_record(self):
        if not self._headless:
            recording_toggle = find_widget_by_tag(self.ui, "recording_toggle")
        # Ensure the recording path ends with .wav
        self.recording_path = self.recording_path
        if self.recording:
            self.graph.start_recording(self.recording_path)
            if not self._headless:
                recording_toggle.style.text_color = 'red'
        else:
            self.graph.stop_recording()
            if not self._headless:
                recording_toggle.style.text_color = 'black'

    def set_master_volume(self):
        self.master_slider_db.set_value(self.master_volume)


class Mapper():
    """Map between two buffers. Typically from a feature buffer to a parameter buffer."""
    def __init__(
            self, 
            source: Feature, 
            target: Union[Dict, List[Dict]],
            in_low: Optional[Union[int, float, List[Union[int, float]]]] = None,
            in_high: Optional[Union[int, float, List[Union[int, float]]]] = None,
            out_low: Optional[Union[int, float, List[Union[int, float]]]] = None,
            out_high: Optional[Union[int, float, List[Union[int, float]]]] = None,
            exponent: Optional[Union[int, float, List[Union[int, float]]]] = 1.0,
            clamp: bool = True,
            name: str = "Mapper"
    ):
        self.name = name
        self.source = source
        self.targets = target if isinstance(target, list) else [target]
        self.num_targets = len(self.targets)

        # save scaling parameters (through setters)
        self.in_low = in_low
        self.in_high = in_high
        self.out_low = out_low
        self.out_high = out_high
        self.exponent = exponent
        self._clamp = clamp

        self.id = str(id(self))

        card_to_name = [target["name"] for target in self.targets]

        self._ui = MapperCard(
            name=self.name,
            id=self.id,
            from_name=self.source.name,
            to_name=card_to_name,
        )
        self._ui.mapper = self

        self._nrt = False
        self._app = None

    @property
    def exponent(self):
        return self._exponent
    
    @exponent.setter
    def exponent(self, value):
        if isinstance(value, float):
            self._exponent = value
        elif isinstance(value, int):
            self._exponent = float(value)
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("exponent must be a number or a list of numbers")
            assert len(value) == self.num_targets, "exponent must have the same length as the number of targets"
            self._exponent = [float(v) for v in value]
        else:
            raise TypeError("exponent must be a number or a list of numbers")

    @property
    def clamp(self):
        return self._clamp
    
    @clamp.setter
    def clamp(self, value):
        self._clamp = value

    @property
    def source_buffer(self):
        # for now only expect Feature objects as the source
        if isinstance(self.source, Feature):
            return self.source.features
        else:
            raise TypeError("Input object must be a Feature")

    @property
    def nrt(self):
        return self._nrt
    
    @nrt.setter
    def nrt(self, value):
        self._nrt = value
        # if switched on,
        if value:
            # create a list for output buffers
            self._output_buffers = []
            # loop throuh all targets
            for i in range(self.num_targets):
                # get the target synth
                target_synth = self.targets[i]["owner"]
                # create output buffer
                _output_buffer = sf.Buffer(target_synth.num_channels, self._app._render_nframes)
                _output_buffer.sample_rate = self._app.fps
                self._output_buffers.append(_output_buffer)
                # set target synth's buffer player to the new buffer
                self.targets[i]["buffer_player"].set_buffer("buffer", _output_buffer)
        # if switched off,
        else:
            # loop through all targets
            for i in range(self.num_targets):
                # set target synth's buffer player back to its internal param buffer
                self.targets[i]["buffer_player"].set_buffer("buffer", self.targets[i]["buffer"])


    @property
    def ui(self):
        return self._ui()

    def __repr__(self):
        return f"Mapper {self.id}: {self.source.name} -> {[target['name'] + '; ' for target in self.targets]}" # show all targets

    @property
    def in_low(self):
        if self._in_low is None:
            if isinstance(self.source, Feature):
                return self.source.min
        else:
            return self._in_low
        
    @in_low.setter
    def in_low(self, value):
        if value is None:
            self._in_low = None
        elif isinstance(value, (int, float)):
            self._in_low = np.array([value]).astype(np.float64)
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("in_low must be a number or a list of numbers")
            self._in_low = np.array(value).astype(np.float64)
        else:
            raise TypeError("in_low must be a number or a list of numbers")
    
    @property
    def in_high(self):
        if self._in_high is None:
            if isinstance(self.source, Feature):
                return self.source.max
        else:
            return self._in_high
        
    @in_high.setter
    def in_high(self, value):
        if value is None:
            self._in_high = None
        elif isinstance(value, (int, float)):
            self._in_high = np.array([value]).astype(np.float64)
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("in_high must be a number or a list of numbers")
            self._in_high = np.array(value).astype(np.float64)
        else:
            raise TypeError("in_high must be a number or a list of numbers")

    @property
    def out_low(self):
        if self._out_low is None:
            return [target["min"] for target in self.targets]
        else:
            return self._out_low
        
    @out_low.setter
    def out_low(self, value):
        if value is None:
            self._out_low = None
        elif isinstance(value, (int, float)):
            self._out_low = np.array([value]).astype(np.float64)
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("out_low must be a number or a list of numbers")
            assert len(value) == self.num_targets, "out_low must have the same length as the number of targets"
            self._out_low = [np.array([v]).astype(np.float64) for v in value]
        else:
            raise TypeError("out_low must be a number or a list of numbers")

    @property
    def out_high(self):
        if self._out_high is None:
            return [target["max"] for target in self.targets]
        else:
            return self._out_high
        
    @out_high.setter
    def out_high(self, value):
        if value is None:
            self._out_high = None
        elif isinstance(value, (int, float)):
            self._out_high = np.array([value]).astype(np.float64)
        elif isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("out_high must be a number or a list of numbers")
            assert len(value) == self.num_targets, "out_high must have the same length as the number of targets"
            self._out_high = [np.array([v]).astype(np.float64) for v in value]
        else:
            raise TypeError("out_high must be a number or a list of numbers")

    def project_to_channels(self, in_data: np.ndarray, num_channels: int) -> np.ndarray:
        in_data_resized = in_data.astype(np.float64)
        if in_data_resized.shape[0] != num_channels:
            in_data_resized = resize_interp(in_data_resized.flatten(), num_channels)
            in_data_resized = in_data_resized.reshape(num_channels, 1)
        return in_data_resized


    def map(self, in_data: np.ndarray) -> List[np.ndarray]:
        """
        Map the source buffer's data (in_data) to all target Synth parameter buffers.
        The default function will also resize-interpolate in_data (with pixasonics.utils.resize_interp) 
        to each target Synth's number of channels before performing the mapping
        (with pixasonics.utils.scale_array_exp).
        For custom mapping schemes, this function can be overridden in subclasses.
        The function should return a list of numpy arrays, one for each target Synth.
        Each numpy array should have a shape of (channels, 1), where channels is the 
        number of channels of the target Synth.

        Args:
            in_data (np.ndarray): The data fetched from the source Feature buffer. This is typically a 2D numpy array of shape (num_features, 1).

        Returns:
            List[np.ndarray]: A list of numpy arrays, one for each target Synth. Each numpy array should have a shape of (channels, 1), where channels is the number of channels of the target Synth.
        """

        # loop through targets
        out_data = []
        for i in range(self.num_targets):
            # resize interpolate in_data, in_low and in_high to the target Synth's number of channels
            target_num_channels = self.targets[i]["owner"].num_channels
            in_data_resized = self.project_to_channels(in_data, target_num_channels)
            in_low = self.project_to_channels(self.in_low, target_num_channels)
            in_high = self.project_to_channels(self.in_high, target_num_channels)
            # fetch the corresponding output range bounds and scaling exponent (if they are lists)
            out_low = self.out_low[i] if isinstance(self.out_low, list) else self.out_low
            out_high = self.out_high[i] if isinstance(self.out_high, list) else self.out_high
            exponent = self.exponent[i] if isinstance(self.exponent, list) else self.exponent
            # scale the input buffer to the output buffer
            scaled_val = scale_array_exp(
                in_data_resized,
                in_low,
                in_high,
                out_low,
                out_high,
                exponent
            ) # shape: (num_channels, 1)
            out_data.append(scaled_val)
        # return the list of scaled values
        return out_data

    def clamp_mappings(self, mappings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clamp the mappings to the output range.
        This is a convenience function that can be used to clamp the mappings
        to the output range defined by out_low and out_high.
        This is called after self.map() with its results.

        Args:
            mappings (List[np.ndarray]): The list of mappings to clamp.

        Returns:
            List[np.ndarray]: The clamped mappings.
        """
        clamped_mappings = []
        for i, mapping in enumerate(mappings):
            out_low = self.out_low[i] if isinstance(self.out_low, list) else self.out_low
            out_high = self.out_high[i] if isinstance(self.out_high, list) else self.out_high
            clamped_mapping = np.clip(mapping, out_low, out_high)
            clamped_mappings.append(clamped_mapping)
        return clamped_mappings
    

    def _map(self, frame=None):
        # get the feature data from its buffer
        in_data = self.source_buffer.data
        mappings = self.map(in_data)

        if self.clamp:
            mappings = self.clamp_mappings(mappings)

        if not self.nrt:
            # if we are in real-time mode, set all targets parameters to the scaled values
            for i in range(self.num_targets):
                scaled_val = mappings[i]
                target_synth = self.targets[i]["owner"]
                # set the parameter to the scaled value
                target_synth.set_input(
                    self.targets[i]["param_name"],
                    scaled_val,
                    from_slider=False
                )
        else:
            # if we are in NRT mode, record the mappings to the output buffers
            for i in range(self.num_targets):
                scaled_val = mappings[i]
                target_output_buffer = self._output_buffers[i]
                target_output_buffer.data[:, frame] = scaled_val[:, 0]

    def __call__(self, frame=None):
        self._map(frame)
