from .features import Feature
from .utils import scale_array_exp, frame2sec, sec2frame, resize_interp, samps2mix
from .ui import MapperCard, AppUI, ImageSettings, ProbeSettings, AudioSettings, Model, find_widget_by_tag
from .synths import Envelope
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
            nrt: bool = False
            ):
        
        self.image_size = image_size


        # Global state variables
        self.is_drawing = False
        self.last_draw_time = time.time()
        self.bg_hires = np.zeros(image_size + (3,), dtype=np.float64)
        self.bg_display = np.zeros(image_size + (3,), dtype=np.uint8)

        # Private properties
        self._fps = fps
        self._refresh_interval = 1 / fps
        self._probe_x = 0
        self._probe_y = 0
        self._mouse_btn = 0
        self._probe_width = Model(50)
        self._probe_height = Model(50)
        self._probe_follows_idle_mouse = Model(True)
        self._interaction_mode = Model("Hold")
        self._last_mouse_down_time = 0
        self._master_volume = Model(0)
        self._audio = Model(False)
        self._recording = Model(False)
        self._recording_path = Model("recording.wav")
        self._unmuted = False
        self._nrt = nrt
        self._normalize_display = Model(False)
        self._normalize_display_global = Model(False)
        self._display_channel_offset = Model(0)
        self._display_layer_offset = Model(0)
        self._image_is_loaded = False

        # Containers for features, mappers, and synths
        self.features = []
        self.mappers = []
        self.synths = []

        self.create_ui()
        self.create_audio_graph()

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
    def nrt(self):
        return self._nrt
    
    @nrt.setter
    def nrt(self, value):
        self._nrt = value
        self.create_audio_graph()
        for mapper in self.mappers:
            mapper.nrt = value

    @property
    def probe_follows_idle_mouse(self):
        return self._probe_follows_idle_mouse.value
    
    @probe_follows_idle_mouse.setter
    def probe_follows_idle_mouse(self, value):
        self._probe_follows_idle_mouse.value = value

    @property
    def probe_x(self):
        return self._probe_x
    
    @probe_x.setter
    def probe_x(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        x_clamped = np.clip(value, self.probe_width//2, self.image_size[1]-1-self.probe_width//2)
        self._probe_x = int(round(x_clamped))
        if not self._nrt:
            self.draw()
    
    @property
    def probe_y(self):
        return self._probe_y
    
    @probe_y.setter
    def probe_y(self, value):
        # clamp to the image size and also no less than half of the probe sides, so that the mouse is always in the middle of the probe
        y_clamped = np.clip(value, self.probe_height//2, self.image_size[0]-1-self.probe_height//2)
        self._probe_y = int(round(y_clamped))
        if not self._nrt:
            self.draw()

    def update_probe_xy(self):
        self.probe_x = self.probe_x
        self.probe_y = self.probe_y

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

    def enable_dsp(self, state: bool):
        self.master_envelope.on() if state else self.master_envelope.off()

    @property
    def interaction_mode(self):
        return self._interaction_mode.value
    
    @interaction_mode.setter
    def interaction_mode(self, value):
        self._interaction_mode.value = value.capitalize()
    

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
        # Canvas keyboard event listeners
        # self.canvas.on_key_down(self.key_callback)

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
        self.graph = sf.AudioGraph.get_shared_graph()
        output_device = sf.AudioOut_Dummy(2) if self.nrt else None
        if self.graph is not None:
            self.graph.destroy()
        self.graph = sf.AudioGraph(
            start=self.nrt,
            output_device=output_device
            )

        # DSP switch
        self.dsp_switch_buf = sf.Buffer(1, 1)
        self.dsp_switch_buf.data[0][0] = 1 if self.nrt else 0
        self.dsp_switch = sf.BufferPlayer(self.dsp_switch_buf, loop=True)

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
        )
        self.master_envelope_bus = sf.Bus(1)
        self.master_envelope_bus.add_input(self.master_envelope.output)
        # add ui to the app ui
        audio_settings = find_widget_by_tag(self.ui, "audio_settings")
        # always keep the first 2 children only, and replace the rest with this env ui
        audio_settings.children = [*audio_settings.children[:2], self.master_envelope.ui]

        # Main bus
        self.bus = sf.Bus(num_channels=2)
        self.audio_out = self.bus * self.master_volume_smooth * self.master_envelope_bus

        # Check if HW has 2 channels
        if self.graph.num_output_channels < 2:
            self.audio_out = sf.ChannelMixer(1, self.audio_out)

        # Add any registered synths to the bus
        for synth in self.synths:
            self.bus.add_input(synth.output)

        # Connect the audio graph
        self.audio_out.play()

        # in NRT mode, unmute the global envelope
        if self.nrt:
            self.master_envelope.on()

        # start graph if audio is enabled
        if self.audio > 0 and not self.nrt:
            self.graph.start()

    
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
            feature.app = self

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
            mapper._app = self

    def detach_mapper(self, mapper):
        print(f"Detaching {mapper}")
        if mapper in self.mappers:
            self.mappers.remove(mapper)
            mappers_carousel = find_widget_by_tag(self.ui, "mappers_carousel")
            mappers_carousel.children = [child for child in mappers_carousel.children if child.tag != f"mapper_{mapper.id}"]
            mapper._ui.app = None
            mapper._app = None
    
    def compute_features(self, probe_mat):
        for feature in self.features:
            feature(probe_mat)
        
    def compute_mappers(self, frame=None):
        for mapper in self.mappers:
            mapper(frame)
        

    def load_image_file(self, image_path):
        img = Image.open(image_path)
        if img.size != self.image_size:
            print("Resizing image to", self.image_size)
            img = img.resize(self.image_size[::-1]) # PIL uses (W, H) instead of (H, W)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[..., None, None] # add channel and layer dimensions if single-channel
        elif len(img.shape) == 3:
            img = img[..., None] # add layer dimension if 3-channel
        print ("Image shape:", img.shape)
        self.bg_hires = img
        self.bg_display = self.convert_image_data_for_display(
            self.bg_hires, 
            normalize=self.normalize_display, 
            global_normalize=self.normalize_display_global)
        
        self._image_is_loaded = True

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

        # Put the image to the canvas
        # img_data = (self.bg_hires * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        # self.canvas[0].put_image_data(self.bg_display, 0, 0)
        self.redraw_background()

        # re-trigger image processing in already attached features
        for feature in self.features:
            feature.app = self


    def load_image_data(self, img_data):
        if img_data.shape[0:2] != self.image_size:
            print("Resizing image data to", self.image_size)
            img_data = self.resize_image_data(img_data)
        self.bg_hires = img_data
        self.bg_display = self.convert_image_data_for_display(
            self.bg_hires, 
            normalize=self.normalize_display, 
            global_normalize=self.normalize_display_global)
        
        self._image_is_loaded = True

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
        for feature in self.features:
            feature.app = self

    
    def resize_image_data(self, img_data):
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


    def get_probe_matrix(self):
        """Get the probe matrix from the background image."""
        x_from = max(self.probe_x - self.probe_width//2, 0)
        y_from = max(self.probe_y - self.probe_height//2, 0)
        probe = self.bg_hires[y_from : y_from + self.probe_height, x_from : x_from + self.probe_width]
        return probe
    

    def render_timeline_to_array(self, timeline):
        out_buf = self.render_timeline(timeline)
        arr = np.copy(out_buf.data)
        self.nrt = 0
        return arr
    

    def render_timeline_to_file(self, timeline, target_filename):
        out_buf = self.render_timeline(timeline)
        out_buf.save(target_filename)
        self.nrt = 0
    

    def render_timeline(self, timeline):
        # create an output buffer to store the rendered audio
        last_time_s, _ = timeline[-1]
        self._output_samps = int(np.ceil(last_time_s * self.graph.sample_rate))
        self._render_nframes = int(np.ceil(last_time_s * self.fps))
        _output_buffer = sf.Buffer(2, self._output_samps)
        _output_buffer.sample_rate = self.graph.sample_rate

        # generate internal timeline buffers for each render frame
        timeline_frames = self.generate_timeline_frames(timeline, self._render_nframes, self.fps)
        
        # switch on NRT mode
        self.nrt = 1

        # render the timeline
        for frame in range(self._render_nframes):
            frame_settings = {key: val[frame] for key, val in timeline_frames.items()}
            self.render_frame(frame, frame_settings)

        # render NRT audio
        self.graph.render_to_buffer(_output_buffer)

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
        for time, settings in timeline:
            new_settings = {**latest_setting, **settings}
            new_timeline.append((time, new_settings))
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

        # Get probe matrix
        probe_mat = self.get_probe_matrix()

        # Compute probe features
        self.compute_features(probe_mat)

        # Update mappings
        if self.unmuted:
            self.compute_mappers()

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
            # Update probe position
            self.probe_x, self.probe_y = x, y
            if pressed == 2:
                if current_time - self._last_mouse_down_time < 0.2:
                    self.mouse_btn = 2 # Double-click
                else:
                    self.mouse_btn = 1 # Single-click
                self._last_mouse_down_time = current_time
            elif pressed == 3:
                self.mouse_btn = 0
            # Update probe features, mappers, and render canvas
            self.draw()

    def key_callback(self, key, shift_key, ctrl_key, meta_key):
        print("Keyboard event:", key, shift_key, ctrl_key, meta_key)


    # GUI callbacks

    def toggle_dsp(self):
        audio_switch = find_widget_by_tag(self.ui, "audio_switch")
        if self.audio:
            self.graph.start()
            audio_switch.style.text_color = 'green'
        else:
            self.graph.stop()
            audio_switch.style.text_color = 'black'

    def toggle_record(self):
        recording_toggle = find_widget_by_tag(self.ui, "recording_toggle")
        # Ensure the recording path ends with .wav
        self.recording_path = self.recording_path
        if self.recording:
            self.graph.start_recording(self.recording_path)
            recording_toggle.style.text_color = 'red'
        else:
            self.graph.stop_recording()
            recording_toggle.style.text_color = 'black'

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

        # expecting a synth's param dict here
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

        self._nrt = False
        self._app = None

    @property
    def buf_in(self):
        # if the input object is an instance of a feature, then we want to map the output of the feature
        # to the input of the object
        if isinstance(self.obj_in, Feature):
            return self.obj_in.features
        # elif isinstance(self.obj_in, dict):
        #     self.buf_in = self.obj_in["buffer"]
        else:
            raise ValueError("Input object must be a Feature")

    @property
    def nrt(self):
        return self._nrt
    
    @nrt.setter
    def nrt(self, value):
        self._nrt = value
        # if switched on,
        if value:
            # create output buffer
            self._output_buffer = sf.Buffer(self.obj_out_owner.num_channels, self._app._render_nframes)
            self._output_buffer.sample_rate = self._app.fps
            # set target synth's buffer player to the new buffer
            self.obj_out["buffer_player"].set_buffer("buffer", self._output_buffer)
        # if switched off,
        else:
            # set target synth's buffer player back to its internal param buffer
            self.obj_out["buffer_player"].set_buffer("buffer", self.obj_out["buffer"])


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


    def map(self, frame=None):
        if self.buf_in.data.shape[0] != self.obj_out_owner.num_channels:
            in_data = resize_interp(self.buf_in.data.flatten(), self.obj_out_owner.num_channels)
            in_data = in_data.reshape(self.obj_out_owner.num_channels, 1)
            in_low = resize_interp(self.in_low.flatten(), self.obj_out_owner.num_channels)
            in_low = in_low.reshape(self.obj_out_owner.num_channels, 1)
            in_high = resize_interp(self.in_high.flatten(), self.obj_out_owner.num_channels)
            in_high = in_high.reshape(self.obj_out_owner.num_channels, 1)
            scaled_val = scale_array_exp(
                in_data,
                in_low,
                in_high,
                self.out_low,
                self.out_high,
                self.exponent
            ) # shape: (num_features, 1)
        else:
            # scale the input buffer to the output buffer
            scaled_val = scale_array_exp(
                self.buf_in.data,
                self.in_low,
                self.in_high,
                self.out_low,
                self.out_high,
                self.exponent
            ) # shape: (num_features, 1)

        if self.clamp:
            scaled_val = np.clip(scaled_val, self.out_low, self.out_high)

        if not self.nrt:
            self.obj_out_owner.set_input_buf(
                self.obj_out["param_name"],
                scaled_val,
                from_slider=False
            )
        else:
            self._output_buffer.data[:, frame] = scaled_val[:, 0]

    def __call__(self, frame=None):
        self.map(frame)
