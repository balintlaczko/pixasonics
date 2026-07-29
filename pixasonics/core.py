from .features import Feature
from .utils import scale_array_exp, sec2frame, resize_interp, samps2mix, ids2mask_cropped, array2str
from .ui import MapperCard, AppUI, DisplaySettings, ProbeSettings, AudioSettings, AudioIOSettingsCard, Model, find_widget_by_tag
from .synths import Synth, Envelope, SynthRegistry
from ipycanvas import hold_canvas, MultiCanvas, Canvas
from IPython.display import display
import time
import numpy as np
import signalflow as sf
from PIL import Image
import threading
from typing import List, Tuple, Dict, Optional, Union, Literal
from functools import lru_cache, wraps
from contextlib import contextmanager
import asyncio
from itertools import chain
import platform


class AudioIOSettings(): # singleton
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioIOSettings, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.graph = sf.AudioGraph.get_shared_graph()
        if self.graph is None:
            self.graph = sf.AudioGraph(start=True)

        self.sample_rate_options = [16000, 22050, 24000, 44100, 48000, 88200, 96000]
        self.buffer_size_options = [16, 32, 64, 128, 256, 441, 480, 512, 1024, 2048, 4096]
        self.card = None

        # create safe defaults for all three platforms (Mac, Windows, Linux) for the most common audio interfaces
        # these will assume English system language---fall back to the first available device if the expected device is not found
        if platform.system() == "Darwin":
            self._backend = "coreaudio" if "coreaudio" in self.backend_names else self.backend_names[0]
            self._input_device = "MacBook Pro Microphone" if "MacBook Pro Microphone" in self.input_device_names else self.input_device_names[0]
            self._output_device = "MacBook Pro Speakers" if "MacBook Pro Speakers" in self.output_device_names else self.output_device_names[0]
        elif platform.system() == "Windows":
            self._backend = "wasapi" if "wasapi" in self.backend_names else self.backend_names[0]
            self._input_device = "Microphone (Realtek(R) Audio)" if "Microphone (Realtek(R) Audio)" in self.input_device_names else self.input_device_names[0]
            self._output_device = "Speakers (Realtek(R) Audio)" if "Speakers (Realtek(R) Audio)" in self.output_device_names else self.output_device_names[0]
        elif platform.system() == "Linux":
            self._backend = "alsa" if "alsa" in self.backend_names else self.backend_names[0]
            self._input_device = "default" if "default" in self.input_device_names else self.input_device_names[0]
            self._output_device = "default" if "default" in self.output_device_names else self.output_device_names[0]
        else:
            raise ValueError(f"Unsupported platform: {platform.system()}")

        self._backend = Model(self._backend)
        self._input_device = Model(self._input_device)
        self._output_device = Model(self._output_device)
        self._sample_rate = Model(48000)
        self._buffer_size = Model(256)

        self.create_ui()
        self.__initialized = True

    def _set_create_graph_button_warning(self):
        create_graph_button = find_widget_by_tag(self.ui, "create_graph_button")
        create_graph_button.button_style = 'warning'

    @property
    def backend_names(self):
        return self.graph.get_backend_names()

    @property
    def input_device_names(self):
        return self.graph.get_input_device_names()

    @property
    def output_device_names(self):
        return self.graph.get_output_device_names()

    @property
    def backend(self):
        return self._backend.value

    @backend.setter
    def backend(self, value):
        if value not in self.backend_names:
            raise ValueError(f"Invalid backend: {value}. Must be one of {self.backend_names}")
        self._backend.value = value
        self._set_create_graph_button_warning()

    @property
    def input_device(self):
        return self._input_device.value

    @input_device.setter
    def input_device(self, value):
        if value not in self.input_device_names:
            raise ValueError(f"Invalid input device: {value}. Must be one of {self.input_device_names}")
        self._input_device.value = value
        self._set_create_graph_button_warning()

    @property
    def output_device(self):
        return self._output_device.value

    @output_device.setter
    def output_device(self, value):
        if value not in self.output_device_names:
            raise ValueError(f"Invalid output device: {value}. Must be one of {self.output_device_names}")
        self._output_device.value = value
        self._set_create_graph_button_warning()

    @property
    def sample_rate(self):
        return self._sample_rate.value

    @sample_rate.setter
    def sample_rate(self, value):
        if value not in self.sample_rate_options:
            raise ValueError(f"Invalid sample rate: {value}. Must be one of {self.sample_rate_options}")
        self._sample_rate.value = value
        self._set_create_graph_button_warning()

    @property
    def buffer_size(self):
        return self._buffer_size.value

    @buffer_size.setter
    def buffer_size(self, value):
        if value not in self.buffer_size_options:
            raise ValueError(f"Invalid buffer size: {value}. Must be one of {self.buffer_size_options}")
        self._buffer_size.value = value
        self._set_create_graph_button_warning()

    @property
    def ui(self):
        return self._ui()

    def get_config(self):
        config = sf.AudioGraphConfig()
        config.output_backend_name = self._backend.value
        config.input_device_name = self._input_device.value
        config.output_device_name = self._output_device.value
        config.sample_rate = self._sample_rate.value
        config.input_buffer_size = self._buffer_size.value
        config.output_buffer_size = self._buffer_size.value
        return config

    def create_audio_graph(self):
        config = self.get_config()
        self.graph = sf.AudioGraph.get_shared_graph()
        if self.graph is not None:
            self.graph.destroy()
        self.graph = sf.AudioGraph(config=config, start=True)

        # check if the graph was created successfully
        if sf.AudioGraph.get_shared_graph() is None:
            raise RuntimeError("Failed to create audio graph. Please check your audio settings and try again.")

        # first, notify all registered synths to re-register themselves with the new graph
        SynthRegistry().notify_reregister(self)
        # then, notify all registered apps to re-create their audio graphs
        AppRegistry().notify_reregister(self)

        # change the color of the create graph button to indicate that the graph has been created
        create_graph_button = find_widget_by_tag(self.ui, "create_graph_button")
        create_graph_button.button_style = 'success'

    def refresh_devices(self):
        """Update the dropdowns with the current devices and backend."""

        backend_dropdown = find_widget_by_tag(self.ui, "backend_dropdown")
        input_device_dropdown = find_widget_by_tag(self.ui, "input_device_dropdown")
        output_device_dropdown = find_widget_by_tag(self.ui, "output_device_dropdown")

        # Update the options and valuesfor each dropdown
        # since at this point the dropdowns are likely aready been bound, we have to first save the current property values, then update the options, and finally restore the property values to the dropdowns
        current_backend = self._backend.value
        current_input_device = self._input_device.value
        current_output_device = self._output_device.value

        backend_dropdown.options = self.backend_names
        input_device_dropdown.options = self.input_device_names
        output_device_dropdown.options = self.output_device_names

        self._backend.value = current_backend if current_backend in self.backend_names else self.backend_names[0]
        self._input_device.value = current_input_device if current_input_device in self.input_device_names else self.input_device_names[0]
        self._output_device.value = current_output_device if current_output_device in self.output_device_names else self.output_device_names[0]

        backend_dropdown.value = self._backend.value
        input_device_dropdown.value = self._input_device.value
        output_device_dropdown.value = self._output_device.value


    def create_ui(self):
        self._ui = AudioIOSettingsCard(
            backend_names=self.backend_names,
            input_device_names=self.input_device_names,
            output_device_names=self.output_device_names,
            sample_rate_options=self.sample_rate_options,
            buffer_size_options=self.buffer_size_options
        )

        # set parent ref
        self._ui.audio_io_settings = self

        # bind props to the dropdowns
        backend_dropdown = find_widget_by_tag(self.ui, "backend_dropdown")
        backend_dropdown.value = self._backend.value
        self._backend.bind_widget(backend_dropdown, extra_callback=self._set_create_graph_button_warning)

        input_device_dropdown = find_widget_by_tag(self.ui, "input_device_dropdown")
        input_device_dropdown.value = self._input_device.value if self._input_device.value in self.input_device_names else self.input_device_names[0]
        self._input_device.bind_widget(input_device_dropdown, extra_callback=self._set_create_graph_button_warning)

        output_device_dropdown = find_widget_by_tag(self.ui, "output_device_dropdown")
        output_device_dropdown.value = self._output_device.value if self._output_device.value in self.output_device_names else self.output_device_names[0]
        self._output_device.bind_widget(output_device_dropdown, extra_callback=self._set_create_graph_button_warning)

        sample_rate_dropdown = find_widget_by_tag(self.ui, "sample_rate_dropdown")
        sample_rate_dropdown.value = self._sample_rate.value if self._sample_rate.value in self.sample_rate_options else self.sample_rate_options[0]
        self._sample_rate.bind_widget(sample_rate_dropdown, extra_callback=self._set_create_graph_button_warning)

        buffer_size_dropdown = find_widget_by_tag(self.ui, "buffer_size_dropdown")
        buffer_size_dropdown.value = self._buffer_size.value if self._buffer_size.value in self.buffer_size_options else self.buffer_size_options[0]
        self._buffer_size.bind_widget(buffer_size_dropdown, extra_callback=self._set_create_graph_button_warning)


class AppRegistry: # singleton
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


def ui_callback(func):
    """
    Decorator for methods that update the UI.
    
    Queues the specific method call (with its arguments) if UI is held.
    """
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        if instance._headless:
            return

        if instance._hold_ui:
            # Create a hashable representation of the call
            # kwargs are converted to a frozenset of items to be hashable
            pending_call = (wrapper, args, frozenset(kwargs.items()))
            instance._pending_ui_callbacks.add(pending_call)
        else:
            return func(instance, *args, **kwargs)
    return wrapper


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

        # Debounced UI flushing (main/UI thread only) — for Features and Synths
        self._ui_flush_handle = None  # asyncio.Handle for scheduled flush
        self._ui_flush_delay = 0.05  # seconds
        self._ui_flush_min_interval = 0.1  # seconds
        self._last_ui_update_time = 0.0  # timestamp of last flush
        self._ui_loop = None  # will hold the UI asyncio loop
        self._ui_widget_refs = {} # references to (frequently updated) UI widgets (to avoid repeated DOM searches)

        # Global state variables
        self.is_drawing = False
        self.last_draw_time = time.perf_counter()
        self.bg_hires = np.zeros(image_size + (1, 1), dtype=np.float64)
        self.bg_display = np.zeros(image_size + (3,), dtype=np.uint8)
        self.mask = np.zeros(image_size + (1, 1), dtype=np.int64)
        self.mask_display = np.zeros(image_size + (3,), dtype=np.uint8)

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
        self._probe_box_select_start_xy = None
        self._probe_follows_idle_mouse = Model(False)
        self._probe_with_mask = Model(False)
        self._same_mask_ids_across_channels = Model(True)
        self._same_mask_ids_across_layers = Model(True)
        self._selected_mask_ids = Model(np.zeros((1, 1)))
        self._interaction_mode = Model("Hold")
        self._last_mouse_down_time = 0
        self._master_volume = Model(0)
        self._audio = Model(False)
        self._recording = Model(False)
        self._recording_path = Model("recording.wav")
        self._unmuted = False
        self._unmuted_on_last_draw = False
        self._nrt = nrt
        self._normalize_display = Model(False)
        self._normalize_display_global = Model(False)
        self._display_normalization_low_percentile = Model([0.0])
        self._display_normalization_high_percentile = Model([100.0])
        self._maximum_displayed_channels = Model(3)
        self._display_channel_offset = Model(0)
        self._display_layer_offset = Model(0)
        self._filter_probe_by_channel_offset = Model(False)
        self._filter_probe_by_layer_offset = Model(False)
        self._image_is_loaded = False
        self._mask_is_loaded = False
        self._mask_channels = -1
        self._mask_layers = -1
        self._highest_mask_id = 0
        self._displayed_mask_bbox = (0, 0, 0, 0) # y1, x1, y2, x2
        self._cached_mask_result = None
        self._headless = headless
        self._draw_lock = False # True disables self.draw()
        self._hold_ui = False # True holds all draw calls and ui updates until hold_ui is set to False again
        self._pending_ui_callbacks = set() # Stores pending callback functions when self._hold_ui is True
        self._mask_display_pending = False
        # a fixed call order for the pending callbacks
        self._ui_callback_order = [
            # update UI elements
            self.toggle_audio_btn,
            self._probe_with_mask_callbacks,
            self.update_mask_ids_display,
            self.update_display_normalization_percentile_display,
            # redraw canvases
            self.redraw_background,
            self.redraw_mask_display,
            self.draw,
        ]

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
        AudioIOSettings() # initialize the singleton if it hasn't been already

    # class-level constant for timeline keys (used in NRT timeline generation)
    TIMELINE_KEYS = {
        "probe_width": {
            "type": int,
            "can_interp": True
        },
        "probe_height": {
            "type": int,
            "can_interp": True
        },
        "probe_x": {
            "type": int,
            "can_interp": True
        },
        "probe_y": {
            "type": int,
            "can_interp": True
        },
        "filter_probe_by_channel_offset": {
            "type": bool,
            "can_interp": False
        },
        "filter_probe_by_layer_offset": {
            "type": bool,
            "can_interp": False
        },
        "display_channel_offset": {
            "type": int,
            "can_interp": True
        },
        "display_layer_offset": {
            "type": int,
            "can_interp": True
        },
        "probe_with_mask": {
            "type": bool,
            "can_interp": False
        },
        "selected_mask_ids": {
            "type": None, # this means values won't be cast in the timeline rendering
            "can_interp": False
        },

    }

    # helper method to check types
    @staticmethod
    def _typecheck(name, value, expected_type):
        """Raise TypeError if value is not of expected_type.
        
        Args:
            name: The property name (for the error message).
            value: The value to check.
            expected_type: A type or tuple of types to check against.
        """
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                expected = " or ".join(t.__name__ for t in expected_type)
            else:
                expected = expected_type.__name__
            raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")


    @contextmanager
    def hold_ui(self):
        """A context manager to suppress and queue UI calls within a block."""
        self._hold_ui = True
        try:
            yield
        finally:
            self._hold_ui = False
            
            # Process pending calls based on the method order
            for ordered_method in self._ui_callback_order:
                ordered_func = getattr(ordered_method, "__func__", ordered_method)
                # Find all queued calls that match the current ordered method
                calls_to_make = [
                    call for call in self._pending_ui_callbacks
                    if call[0] == ordered_func
                ]
                
                for call in calls_to_make:
                    func, args, kwargs_frozenset = call
                    # Recreate kwargs dict from frozenset and execute
                    func(self, *args, **dict(kwargs_frozenset))
                    # Remove the call from the main set once executed
                    self._pending_ui_callbacks.remove(call)

            # In case any decorated method was not in the order list, run it now.
            for func, args, kwargs_frozenset in self._pending_ui_callbacks.copy():
                func(self, *args, **dict(kwargs_frozenset))

            self._pending_ui_callbacks.clear()

    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._typecheck("fps", value, int)
        self._fps = value
        self._refresh_interval = 1 / value

    @property
    def normalize_display(self):
        return self._normalize_display.value
    
    @normalize_display.setter
    def normalize_display(self, value):
        self._typecheck("normalize_display", value, bool)
        self._normalize_display.value = value
        self.redraw_background()

    @property
    def normalize_display_global(self):
        return self._normalize_display_global.value
    
    @normalize_display_global.setter
    def normalize_display_global(self, value):
        self._typecheck("normalize_display_global", value, bool)
        self._normalize_display_global.value = value
        self.redraw_background()
    
    def _display_normalization_setter_typecheck(self, value, which: Literal['low', 'high']):
        """Validate and normalize the value for display normalization percentile setters.

        Accepts an int, float, or list of numbers and converts it to a list of floats.
        If a list is provided, its length must be either 1 (applied to all channels)
        or equal to the number of channels in the loaded image.

        Args:
            value: The value to validate. Must be an int, float, or list of numbers.
            which: Which percentile is being set, either 'low' or 'high'. Used in error messages.

        Returns:
            list[float]: The validated value as a list of floats.

        Raises:
            TypeError: If the value is not an int, float, or list.
            ValueError: If the list length is not 1 or equal to the number of image channels.
        """
        # only accept int, float and list
        if isinstance(value, (int, float)):
            value = [float(value)]
        elif isinstance(value, list):
            # check that list length is 1 or equal to number of channels
            if len(value) not in [1, self.bg_hires.shape[2]]:
                raise ValueError(f"Invalid length for display_normalization_{which}_percentile list. Expected 1 or {self.bg_hires.shape[2]}, got {len(value)}.")
            value = [float(v) for v in value] # convert to float
        else:
            raise TypeError(f"Invalid value for display_normalization_{which}_percentile. Expected int, float, or list.")
        return value
    
    @property
    def display_normalization_low_percentile(self):
        return self._display_normalization_low_percentile.value

    @display_normalization_low_percentile.setter
    def display_normalization_low_percentile(self, value):
        # type check: only accept int, float and list
        value = self._display_normalization_setter_typecheck(value, which="low")
        value = [max(0.0, min(100.0, v)) for v in value]  # clamp to [0, 100]
        changed = value != self._display_normalization_low_percentile.value
        self._display_normalization_low_percentile.value = value
        if changed:
            self._display_normalization_percentile_callbacks(which="low")

    @property
    def display_normalization_high_percentile(self):
        return self._display_normalization_high_percentile.value

    @display_normalization_high_percentile.setter
    def display_normalization_high_percentile(self, value):
        # type check: only accept int, float and list
        value = self._display_normalization_setter_typecheck(value, which="high")
        value = [max(0.0, min(100.0, v)) for v in value]  # clamp to [0, 100]
        changed = value != self._display_normalization_high_percentile.value
        self._display_normalization_high_percentile.value = value
        if changed:
            self._display_normalization_percentile_callbacks(which="high")

    def _display_normalization_percentile_callbacks(self, which: Literal['low', 'high']):
        self.update_display_normalization_percentile_display(which=which)
        self.redraw_background()

    @property
    def maximum_displayed_channels(self):
        return self._maximum_displayed_channels.value
    
    @maximum_displayed_channels.setter
    def maximum_displayed_channels(self, value):
        self._typecheck("maximum_displayed_channels", value, int)
        value = int(np.clip(value, 1, 3))
        changed = value != self._maximum_displayed_channels.value
        self._maximum_displayed_channels.value = value
        if changed:
            self.maximum_displayed_channels_callbacks()

    def maximum_displayed_channels_callbacks(self):
        self.redraw_background()
        self.redraw_mask_display()

    @property
    def display_channel_offset(self):
        return self._display_channel_offset.value
    
    @display_channel_offset.setter
    def display_channel_offset(self, value):
        self._typecheck("display_channel_offset", value, int)
        value = int(np.clip(value, 0, max(0, self.bg_hires.shape[2]-1)))
        self._display_channel_offset.value = value
        self._display_channel_or_layer_offset_callbacks()

    @property
    def display_layer_offset(self):
        return self._display_layer_offset.value
    
    @display_layer_offset.setter
    def display_layer_offset(self, value):
        self._typecheck("display_layer_offset", value, int)
        value = int(np.clip(value, 0, max(0, self.bg_hires.shape[3]-1)))
        self._display_layer_offset.value = value
        self._display_channel_or_layer_offset_callbacks()

    def _display_channel_or_layer_offset_callbacks(self):
        if self._nrt:
            return
        if self.filter_probe_by_layer_offset or self.filter_probe_by_channel_offset:
            self.compute()
        self.redraw_background()
        self.redraw_mask_display()

    @property
    def filter_probe_by_channel_offset(self):
        return self._filter_probe_by_channel_offset.value
    
    @filter_probe_by_channel_offset.setter
    def filter_probe_by_channel_offset(self, value):
        self._typecheck("filter_probe_by_channel_offset", value, bool)
        changed = value != self._filter_probe_by_channel_offset.value
        self._filter_probe_by_channel_offset.value = value
        if changed and not self._nrt:
            self.compute()

    @property
    def filter_probe_by_layer_offset(self):
        return self._filter_probe_by_layer_offset.value

    @filter_probe_by_layer_offset.setter
    def filter_probe_by_layer_offset(self, value):
        self._typecheck("filter_probe_by_layer_offset", value, bool)
        changed = value != self._filter_probe_by_layer_offset.value
        self._filter_probe_by_layer_offset.value = value
        if changed and not self._nrt:
            self.compute()

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
        self._typecheck("nrt", value, bool)
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
        self._typecheck("probe_follows_idle_mouse", value, bool)
        self._probe_follows_idle_mouse.value = value

    @property
    def probe_with_mask(self):
        return self._probe_with_mask.value

    @probe_with_mask.setter
    def probe_with_mask(self, value):
        self._typecheck("probe_with_mask", value, bool)
        # if no mask is loaded keep it off
        if not self._mask_is_loaded:
            value = False
        changed = value != self._probe_with_mask.value
        self._probe_with_mask.value = value
        if changed:
            self._cached_mask_result = None  # invalidate cache
        if changed and not self._nrt:
            self._probe_with_mask_callbacks()

    @ui_callback
    def _probe_with_mask_callbacks(self):
            self._selected_mask_ids.value = self.get_mask_ids_under_probe() # don't trigger setter
            self.redraw_mask_display()
            self.update_mask_ids_display()
            self.compute_and_draw()
            # enable/disable selected mask ID numbox on UI
            numbox_selected_mask_id = find_widget_by_tag(self.ui, "selected_mask_id")
            numbox_selected_mask_id.disabled = not self.probe_with_mask
            # enable/disable same IDs across channels checkbox on UI
            checkbox_same_mask_ids_across_channels = find_widget_by_tag(self.ui, "same_mask_ids_across_channels")
            checkbox_same_mask_ids_across_channels.disabled = not self.probe_with_mask
            # enable/disable same IDs across layers checkbox on UI
            checkbox_same_mask_ids_across_layers = find_widget_by_tag(self.ui, "same_mask_ids_across_layers")
            checkbox_same_mask_ids_across_layers.disabled = not self.probe_with_mask
            # enable/disable clear all selections button on UI
            btn_clear_all_selections = find_widget_by_tag(self.ui, "clear_all_selections_btn")
            btn_clear_all_selections.disabled = not self.probe_with_mask

    @property
    def same_mask_ids_across_channels(self):
        return self._same_mask_ids_across_channels.value

    @same_mask_ids_across_channels.setter
    def same_mask_ids_across_channels(self, value):
        self._typecheck("same_mask_ids_across_channels", value, bool)
        self._same_mask_ids_across_channels.value = value
        self._same_mask_ids_across_channels_or_layers_callbacks()

    @property
    def same_mask_ids_across_layers(self):
        return self._same_mask_ids_across_layers.value

    @same_mask_ids_across_layers.setter
    def same_mask_ids_across_layers(self, value):
        self._typecheck("same_mask_ids_across_layers", value, bool)
        self._same_mask_ids_across_layers.value = value

    @property
    def selected_mask_ids(self):
        return self._selected_mask_ids.value

    @selected_mask_ids.setter
    def selected_mask_ids(self, value):
        # ignore if no mask is loaded
        if not self._mask_is_loaded:
            return
        # if None, set to a 2D zero array
        set_to_none = False
        if value is None:
            set_to_none = True
            value = np.zeros((1, 1), dtype= np.uint16)
        # if number convert to a 2d array (channels, layers)
        if np.isscalar(value):
            value = np.array([[int(value)]])
        # if it is a list or tuple, convert to a 1D array
        elif type(value) in [list, tuple]:
            value = np.array(value)
        # if it is a set, convert it to a list first (to have a consistent order)
        elif type(value) == set:
            value = np.array(list(value))
        # if it is an array
        elif type(value) == np.ndarray:
            # 1D, do not allow object type, otherwise fine
            if np.ndim(value) == 1:
                if value.dtype == object:
                    raise ValueError(f"Cannot interpret a 1D object array. For object-type arrays, use a 2D array with shape broadcastable to {(self.mask_channels, self.mask_layers)}.")
            # 0D, convert to 2D
            if np.ndim(value) == 0:
                value = np.array([[value]])
            # if 2D, make sure it can be broadcast to the loaded mask's (channels, layers)
            elif np.ndim(value) == 2:
                try:
                    _test = np.broadcast_to(value, (self.mask_channels, self.mask_layers))
                except ValueError:
                    raise ValueError(f"Invalid array shape. Cannot broadcast to mask shape {(self.mask_channels, self.mask_layers)}")
                # if object type, make sure each element is a list or int
                if value.dtype == object:
                    for elem in value.flat:
                        if not isinstance(elem, (list, int)):
                            raise TypeError(f"Invalid element type in object array. Expected list or int, got {type(elem)}.")
            # do not allow more than 2D
            elif np.ndim(value) > 2:
                raise ValueError(f"Invalid array shape. Expected up-to 2D array, got {np.ndim(value)}D")
        # repeat the previous value if all zeros (if not object type or not deliberately set to None)
        if value.dtype != object and not set_to_none:
            value = value if value.sum() != 0 else self._selected_mask_ids.value
        changed = not np.array_equal(value, self._selected_mask_ids.value)
        self._selected_mask_ids.value = value
        if changed:
            self._cached_mask_result = None  # invalidate cache
        if changed or self._unmuted_changed:
            self._selected_mask_ids_callbacks()

    def _selected_mask_ids_callbacks(self):
        if self._nrt:
            return
        self.update_mask_ids_display()
        self.redraw_mask_display()
        self.compute_and_draw()

    @property
    def _multiple_mask_ids_per_sheet_selected(self):
        selected_mask_ids = self.selected_mask_ids # only get it once
        object_type = selected_mask_ids.dtype == object
        empty = not np.any(selected_mask_ids)
        return selected_mask_ids.ndim == 1 or (selected_mask_ids.ndim == 2 and (object_type and not empty))

    def mouse_within_mask_bbox(self, x, y):
        y1, x1, y2, x2 = self._displayed_mask_bbox
        return x >= x1 and x <= x2 and y >= y1 and y <= y2

    @property
    def highest_mask_id(self):
        return self._highest_mask_id
    
    @property
    def mask_channels(self):
        return self._mask_channels

    @property
    def mask_layers(self):
        return self._mask_layers

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
        self._typecheck("probe_x", value, (int, float))
        value = self.clamp_probe_x(value)
        changed = value != self._probe_x
        if not changed:
            return
        self._probe_x = value
        if self.probe_with_mask and not self._nrt:
            self.selected_mask_ids = self.get_mask_ids_under_probe()
        if not self._nrt:
            self.compute_and_draw()
    
    @property
    def probe_y(self):
        return self._probe_y
    
    @probe_y.setter
    def probe_y(self, value):
        self._typecheck("probe_y", value, (int, float))
        value = self.clamp_probe_y(value)
        changed = value != self._probe_y
        if not changed:
            return
        self._probe_y = value
        if self.probe_with_mask and not self._nrt:
            self.selected_mask_ids = self.get_mask_ids_under_probe()
        if not self._nrt:
            self.compute_and_draw()

    def update_probe_xy(self):
        # Apply the clamped probe position without triggering a draw
        old_probe_x, old_probe_y = self._probe_x, self._probe_y
        self._probe_x = self.clamp_probe_x(self.probe_x)
        self._probe_y = self.clamp_probe_y(self.probe_y)
        changed = (self._probe_x != old_probe_x) or (self._probe_y != old_probe_y)
        if self.probe_with_mask and changed and not self._nrt:
            self.selected_mask_ids = self.get_mask_ids_under_probe()
        if not self._nrt:
            self.compute_and_draw()

    @property
    def mouse_btn(self):
        return self._mouse_btn
    
    @mouse_btn.setter
    def mouse_btn(self, value):
        self._typecheck("mouse_btn", value, int)
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
        self._typecheck("probe_width", value, int)
        self._probe_width.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def probe_height(self):
        return int(self._probe_height.value)
    
    @probe_height.setter
    def probe_height(self, value):
        self._typecheck("probe_height", value, int)
        self._probe_height.value = value
        # Update mouse xy to keep it in the middle of the probe
        self.update_probe_xy()

    @property
    def _probe_changed(self):
        return (
            self._probe_x != self._probe_x_on_last_draw
            or self._probe_y != self._probe_y_on_last_draw
            or self._probe_width.value != self._probe_width_on_last_draw
            or self._probe_height.value != self._probe_height_on_last_draw
        )

    @property
    def master_volume(self):
        return self._master_volume.value
    
    @master_volume.setter
    def master_volume(self, value):
        self._typecheck("master_volume", value, (int, float))
        self._master_volume.value = value
        self.set_master_volume()

    @property
    def audio(self):
        return self._audio.value
    
    @audio.setter
    def audio(self, value):
        self._typecheck("audio", value, bool)
        self._audio.value = value
        self.toggle_dsp()

    @property
    def recording(self):
        return self._recording.value
    
    @recording.setter
    def recording(self, value):
        self._typecheck("recording", value, bool)
        self._recording.value = value
        self.toggle_record()

    @property
    def recording_path(self):
        return self._recording_path.value
    
    @recording_path.setter
    def recording_path(self, value):
        self._typecheck("recording_path", value, str)
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
        self._typecheck("unmuted", value, bool)
        changed = value != self._unmuted
        self._unmuted = value
        if value:
            self.master_envelope.on()
        else:
            self.master_envelope.off()
        if changed and not self._nrt:
            self.redraw_mask_display()
            self.compute_and_draw()


    @property
    def _unmuted_changed(self):
        return self._unmuted != self._unmuted_on_last_draw
    
    @property
    def output_buffer_size(self):
        if self.graph is not None:
            return self.graph.output_buffer_size
        else:
            return None

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
        self._typecheck("interaction_mode", value, str)
        if value.capitalize() not in ["Hold", "Toggle", "Select"]:
            raise ValueError("Invalid interaction mode. Expected one of: 'Hold', 'Toggle', 'Select'.")
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
            self.compute_event.wait() # block until signaled
            # Drain/coalesce multiple signals that arrived during compute
            while self.compute_event.is_set():
                self.compute_event.clear()
                with self.compute_lock:
                    probe_mat = self.get_probe_matrix()
                    self.compute_features(probe_mat)
                    if self.unmuted:
                        self.compute_mappers()
                # time.sleep(0.01)  # Small sleep to prevent busy-waiting

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
        display_settings = DisplaySettings()
        probe_settings = ProbeSettings(
            canvas_width=self.image_size[1],
            canvas_height=self.image_size[0]
        )
        audio_settings = AudioSettings()
        self.ui = AppUI(
            audio_settings, 
            display_settings, 
            probe_settings,
            canvas_height=self.image_size[0],
            canvas_width=self.image_size[1])()
        display(self.ui)

        # Capture the UI asyncio loop (if available)
        try:
            self._ui_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._ui_loop = None

        # Create the canvas
        self.canvas = MultiCanvas(
            3,
            width=self.image_size[1], 
            height=self.image_size[0])
        app_canvas = find_widget_by_tag(self.ui, "app_canvas")
        app_canvas.children = [self.canvas]

        # Canvas mousing event listeners
        self.canvas.on_mouse_move(lambda x, y: self.mouse_callback(x, y, -1))  # Triggered during mouse movement (keeps track of mouse button state)
        self.canvas.on_mouse_down(lambda x, y: self.mouse_callback(x, y, pressed=2))  # When mouse button pressed
        self.canvas.on_mouse_up(lambda x, y: self.mouse_callback(x, y, pressed=3))  # When mouse button released

        # Bind image settings widgets
        # Normalize display checkboxes
        chkbox_normalize_display = find_widget_by_tag(self.ui, "normalize_display")
        self._normalize_display.bind_widget(chkbox_normalize_display, extra_callback=self.redraw_background)
        chkbox_normalize_display_global = find_widget_by_tag(self.ui, "normalize_display_global")
        self._normalize_display_global.bind_widget(chkbox_normalize_display_global, extra_callback=self.redraw_background)
        # Low Percentile slider
        # need to do it manually because it is not a constant two-way model
        display_normalization_low_percentile_slider = find_widget_by_tag(self.ui, "display_normalization_low_percentile_slider")
        def _update_display_normalization_percentile_from_slider(change, which, extra_callback=None):
            selected_prop_name = f'display_normalization_{which}_percentile'
            selected_prop_val = getattr(self, selected_prop_name)[0]
            if change['name'] == 'value' and change['new'] != selected_prop_val:
                setattr(self, selected_prop_name, change['new'])
                if extra_callback is not None:
                    extra_callback()
        display_normalization_low_percentile_slider.observe(lambda x: _update_display_normalization_percentile_from_slider(x, which='low', extra_callback=self.redraw_background), names='value')
        # High Percentile numbox
        # need to do it manually because it is not a constant two-way model
        display_normalization_high_percentile_slider = find_widget_by_tag(self.ui, "display_normalization_high_percentile_slider")
        display_normalization_high_percentile_slider.observe(lambda x: _update_display_normalization_percentile_from_slider(x, which='high', extra_callback=self.redraw_background), names='value')
        # Maximum displayed channels slider
        maximum_displayed_channels_slider = find_widget_by_tag(self.ui, "maximum_displayed_channels")
        self._maximum_displayed_channels.bind_widget(maximum_displayed_channels_slider, extra_callback=self.maximum_displayed_channels_callbacks)
        # Channel and layer offset sliders
        channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
        self._display_channel_offset.bind_widget(channel_offset_slider, extra_callback=self._display_channel_or_layer_offset_callbacks)
        layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
        self._display_layer_offset.bind_widget(layer_offset_slider, extra_callback=self._display_channel_or_layer_offset_callbacks)

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
        # Filter by channel/layer offset checkboxes
        filter_probe_by_channel_offset = find_widget_by_tag(self.ui, "filter_probe_by_channel_offset")
        self._filter_probe_by_channel_offset.bind_widget(filter_probe_by_channel_offset, extra_callback=self.compute_event.set)
        filter_probe_by_layer_offset = find_widget_by_tag(self.ui, "filter_probe_by_layer_offset")
        self._filter_probe_by_layer_offset.bind_widget(filter_probe_by_layer_offset, extra_callback=self.compute_event.set)
        # Probe with mask checkbox
        chkbox_probe_with_mask = find_widget_by_tag(self.ui, "probe_with_mask")
        self._probe_with_mask.bind_widget(chkbox_probe_with_mask, extra_callback=self._probe_with_mask_callbacks)
        # Same mask IDs across channels checkbox
        chkbox_same_mask_ids_across_channels = find_widget_by_tag(self.ui, "same_mask_ids_across_channels")
        self._same_mask_ids_across_channels.bind_widget(chkbox_same_mask_ids_across_channels)#, extra_callback=self._same_mask_ids_across_channels_or_layers_callbacks)
        # Same mask IDs across layers checkbox
        chkbox_same_mask_ids_across_layers = find_widget_by_tag(self.ui, "same_mask_ids_across_layers")
        self._same_mask_ids_across_layers.bind_widget(chkbox_same_mask_ids_across_layers)#, extra_callback=self._same_mask_ids_across_channels_or_layers_callbacks)
        # Selected mask ID numbox
        # need to do it manually because it is not a constant two-way model
        numbox_selected_mask_id = find_widget_by_tag(self.ui, "selected_mask_id")
        def _update_selected_mask_id_from_numbox(change, extra_callback=None):
            _value = np.array([[int(change['new'])]]) if change['new'] is not None else np.zeros((1, 1), dtype=np.uint16)
            if change['name'] == 'value' and not np.array_equal(_value, self.selected_mask_ids):
                self.selected_mask_ids = change['new'] # it will handle the scalar
                if extra_callback is not None:
                    extra_callback()
        numbox_selected_mask_id.observe(lambda x: _update_selected_mask_id_from_numbox(x, extra_callback=self._selected_mask_ids_callbacks), names='value')
        # Clear all selections button
        btn_clear_all_selections = find_widget_by_tag(self.ui, "clear_all_selections_btn")
        def _clear_all_selections_callback(event):
            self.selected_mask_ids = None
        btn_clear_all_selections.on_click(_clear_all_selections_callback)

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


    # debounced UI flush scheduler for Features and Synths
    def schedule_ui_flush(self, delay: float | None = None):
        if getattr(self, "_headless", False):
            return
        d = self._ui_flush_delay if delay is None else float(delay)

        loop = self._ui_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._ui_loop = loop
            except RuntimeError:
                # No UI loop yet; draw() will still opportunistically flush
                return

        def _schedule():
            # cancel previously scheduled flush (debounce)
            if self._ui_flush_handle is not None:
                try:
                    self._ui_flush_handle.cancel()
                except Exception:
                    pass
            self._ui_flush_handle = loop.call_later(d, self.flush_ui)

        loop.call_soon_threadsafe(_schedule)

    # flush both Feature and Synth UI on the UI loop
    def flush_ui(self):
        try:
            # Features
            for feature in list(getattr(self, "features", [])):
                if getattr(feature, "_pending_ui", False):
                    if hasattr(feature, "update_ui_throttled"):
                        feature.update_ui_throttled(force=True)
                    elif hasattr(feature, "update_ui"):
                        feature.update_ui()
                    feature._pending_ui = False
            # Synths
            for synth in list(getattr(self, "synths", [])):
                if getattr(synth, "_pending_ui", False):
                    if hasattr(synth, "update_ui_throttled"):
                        synth.update_ui_throttled(force=True)
                    elif hasattr(synth, "update_ui"):
                        synth.update_ui()
                    synth._pending_ui = False
        finally:
            self._ui_flush_handle = None


    def __call__(self):
        return self.ui
    

    def create_audio_graph(self):
        # Get or create the shared audio graph
        self.graph = sf.AudioGraph.get_shared_graph()
        if self.graph is None:
            self.graph = AudioIOSettings().graph


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
        # (in case of a new graph, the AudioIOSettings will ask the SynthRegistry to re-register all synths before this step)
        for synth in self.synths:
            self.bus.add_input(synth.output)

        # put the app's graph to the global graph if audio is enabled
        # but not in nrt, because in that case we don't want to play the audio
        if self.audio > 0 and not self.nrt:
            self.audio_out.play()
            self.unmuted = self.unmuted # call the setter to update the envelope state

    
    def attach_synth(self, synth):
        if synth not in self.synths:
            self.synths.append(synth)
            self.bus.add_input(synth.output)
            if not self._headless:
                synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
                synths_carousel.children = list(synths_carousel.children) + [synth.ui]
                synth._ui.app = self
        synth.app = self

    def detach_synth(self, synth):
        if synth in self.synths:
            self.synths.remove(synth)
            self.bus.remove_input(synth.output)
            if not self._headless:
                synths_carousel = find_widget_by_tag(self.ui, "synths_carousel")
                synths_carousel.children = [child for child in synths_carousel.children if child.tag != f"synth_{synth.id}"]
                synth._ui.app = None
        if getattr(synth, "app", None) is self:
            synth.app = None
    
    def attach_feature(self, feature):
        if feature not in self.features:
            self.features.append(feature)
            if not self._headless:
                features_carousel = find_widget_by_tag(self.ui, "features_carousel")
                features_carousel.children = list(features_carousel.children) + [feature.ui]
                feature._ui.app = self
        feature.app = self

    def detach_feature(self, feature):
        if feature in self.features:
            self.features.remove(feature)
            if not self._headless:
                features_carousel = find_widget_by_tag(self.ui, "features_carousel")
                features_carousel.children = [child for child in features_carousel.children if child.tag != f"feature_{feature.id}"]
                feature._ui.app = None
        if getattr(feature, "app", None) is self:
            feature.app = None
    
    def attach_mapper(self, mapper):
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
        """
        Load an image file and update the internal state.
        Uses PIL to open the image, then calls `load_image_data`.

        Args:
            image_path (str): The path to the image file.
            refresh_features (bool, optional): Whether to refresh the currently attached Features after loading the image. Defaults to True.
        """
        img = Image.open(image_path)
        img = np.array(img)
        self.load_image_data(img, refresh_features=refresh_features)


    def load_image_data(self, img_data: np.ndarray, refresh_features: bool = True) -> None:
        """
        Load image data as a NumPy array. The Height and Width dimensions will be resized to match `image_size`.
        The input will be reshaped to 4D: (Height, Width, Channels, Layers)

        Args:
            img_data (np.ndarray): The image data as a NumPy array.
            refresh_features (bool, optional): Whether to refresh the currently attached Features after loading the image. Defaults to True.
        """
        # resize if H/W is not self.image_size
        if img_data.shape[0:2] != self.image_size:
            img_data = self.resize_image_data(img_data)
        # make sure it is 4D
        elif len(img_data.shape) != 4:
            img_data = self.reshape_image_data(img_data)
        self.bg_hires = img_data
        
        self._image_is_loaded = True
        # clear cache
        self._get_rendered_image_display.cache_clear()

        if not self._headless:
            # Set layer offset to 0, enable the slider, and set the max value
            layer_offset_slider = find_widget_by_tag(self.ui, "layer_offset")
            self._display_layer_offset.value = 0
            if len(self.bg_hires.shape) == 4 and self.bg_hires.shape[3] > 1:
                layer_offset_slider.disabled = False
                layer_offset_slider.max = self.bg_hires.shape[-1] - 1
            else:
                layer_offset_slider.disabled = True
                layer_offset_slider.max = 0

            # Set the channel offset to 0, enable the slider, and set the max value
            channel_offset_slider = find_widget_by_tag(self.ui, "channel_offset")
            self._display_channel_offset.value = 0
            if self.bg_hires.shape[2] > 1:
                channel_offset_slider.disabled = False
                channel_offset_slider.max = self.bg_hires.shape[2] - 1
            else:
                channel_offset_slider.disabled = True
                channel_offset_slider.max = 0

            # check display percentile lists, so that they either have 1 element, or a list with length equal to the number of channels
            if len(self.display_normalization_low_percentile) not in [1, self.bg_hires.shape[2]]:
                self.display_normalization_low_percentile = 0
                print(f'Warning: Resetting display normalization low percentile to 0, because the provided list length does not match the number of channels ({self.bg_hires.shape[2]}).')
            if len(self.display_normalization_high_percentile) not in [1, self.bg_hires.shape[2]]:
                self.display_normalization_high_percentile = 100
                print(f'Warning: Resetting display normalization high percentile to 100, because the provided list length does not match the number of channels ({self.bg_hires.shape[2]}).')
            # Redraw the background with the new image
            self.redraw_background()

        # re-trigger image processing in already attached features
        if refresh_features:
            for feature in self.features:
                feature.app = self


    def load_mask_file(self, mask_path: str) -> None:
        """
        Load a mask (image) file.
        Uses PIL to open the image, then calls `load_mask_data`.

        Args:
            mask_path (str): The path to the mask file.
        """
        mask = Image.open(mask_path)
        mask = np.array(mask)
        self.load_mask_data(mask)


    def load_mask_data(self, mask_data: np.ndarray) -> None:
        """
        Load mask data as a NumPy array. The Height and Width dimensions will be resized to match `image_size`.
        The input will be reshaped to 4D: (Height, Width, Channels, Layers)
        Registers that the mask is loaded.

        Args:
            mask_data (np.ndarray): The mask data as a NumPy array.
        """
        # resize if H/W is not self.image_size
        if mask_data.shape[0:2] != self.image_size:
            mask_data = self.resize_image_data(mask_data, use_nearest=True)
        # make sure it is 4D
        elif len(mask_data.shape) != 4:
            mask_data = self.reshape_image_data(mask_data)
        # try broadcasting and raise error if not compatible
        try:
            np.broadcast_to(mask_data, self.bg_hires.shape)
        except ValueError:
            raise ValueError(f"Mask shape {mask_data.shape} is not compatible with image shape {self.bg_hires.shape}")
        self.mask = mask_data
        self._mask_is_loaded = True
        self._mask_channels = self.mask.shape[2]
        self._mask_layers = self.mask.shape[3]
        # clear cache
        self._get_rendered_mask_display.cache_clear()
        # set UI stuff
        if not self._headless:
            # enable Probe with mask checkbox
            chkbox_probe_with_mask = find_widget_by_tag(self.ui, "probe_with_mask")
            chkbox_probe_with_mask.disabled = False
            # set maximum bound to the highest label ID
            self._highest_mask_id = int(self.mask.max())
            numbox_selected_mask_id = find_widget_by_tag(self.ui, "selected_mask_id")
            numbox_selected_mask_id.max = self._highest_mask_id
            # render either a numbox of a text field
            self.update_mask_ids_display()
            # redraw mask display if probing with mask
            if self.probe_with_mask:
                self.selected_mask_ids = self.get_mask_ids_under_probe()
                self.redraw_mask_display()


    def resize_image_data(self, img_data: np.ndarray, use_nearest: bool=False) -> np.ndarray:
        """
        Resize the image data to the App's image_size attribute.

        Args:
            img_data (np.ndarray): The image data to resize.
            use_nearest (bool, optional): Whether to use nearest neighbor interpolation. Defaults to False, which then uses bicubic.

        Returns:
            np.ndarray: The resized image data.
        """
        # masks must use nearest neighbor interpolation
        interp = Image.Resampling.NEAREST if use_nearest else Image.Resampling.BICUBIC
        # make sure it is 4D
        img_data = self.reshape_image_data(img_data)
        # loop through the layers and resize each one
        img_data_resized = np.zeros(self.image_size + img_data.shape[2:], dtype=img_data.dtype)
        for i in range(img_data.shape[3]):
            layer = img_data[:, :, :, i]
            resized_layer = np.zeros(self.image_size + (layer.shape[2],), dtype=img_data.dtype)
            for j in range(layer.shape[2]):
                img = Image.fromarray(layer[:, :, j])
                resized_layer[:, :, j] = np.array(img.resize(self.image_size[::-1], resample=interp)) # PIL uses (W, H) instead of (H, W)
            img_data_resized[:, :, :, i] = resized_layer
        return img_data_resized
    

    def reshape_image_data(self, img_data: np.ndarray) -> np.ndarray:
        """
        Reshape the image data to 4D (H, W, C, L).

        Args:
            img_data (np.ndarray): The image data to reshape.

        Returns:
            np.ndarray: The reshaped image data (always 4D).
        """
        # if 2D, add channel and layer dimensions
        if len(img_data.shape) == 2:
            img_data = img_data[..., None, None]
        # if 3D, add a layer dimension
        elif len(img_data.shape) == 3:
            img_data = img_data[..., None]
        # raise error if 2> or 4<
        elif len(img_data.shape) < 2 or len(img_data.shape) > 4:
            raise ValueError(f"Invalid image data shape, expected 2 to 4 dimensions, got {len(img_data.shape)}")
        return img_data


    def convert_image_data_for_display(
            self, 
            img_data: np.ndarray, 
            normalize: bool = False,
            global_normalize: bool = False,
            norm_low_percentile: Union[float, List[float]] = 0.0,
            norm_high_percentile: Union[float, List[float]] = 100.0,
            maximum_displayed_channels: Literal[1, 2, 3] = 3,
            channel_offset: int = 0,
            layer_offset: int = 0,
            add_transparency: bool = False,
            opacity: float = 0.5
            ):
        img = self.rescale_image_data_for_display(
            img_data, 
            normalize=normalize, 
            global_normalize=global_normalize,
            norm_low_percentile=norm_low_percentile,
            norm_high_percentile=norm_high_percentile
        )
        # if 4D, slice the layer according to the layer offset
        if len(img.shape) == 4:
            img = img[:, :, :, layer_offset]
        # if more than 1ch apply layer offset
        if img.shape[2] > 1:
            img = img[:, :, channel_offset : min(channel_offset + maximum_displayed_channels, img.shape[2])]
        # if (result is) single channel, repeat to 3 channels
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        # if two channels, add a third empty channel
        elif img.shape[2] == 2:
            img = np.concatenate([img, np.zeros(img.shape[:2] + (1,), dtype=img.dtype)], axis=2)
        # if adding transparency, copy the first channel into the alpha and multiply it with the opacity value
        if add_transparency:
            img = np.concatenate([img, (img[..., :1] * opacity)], axis=2)
        return img


    def rescale_image_data_for_display(
            self, 
            img_data: np.ndarray, 
            normalize: bool = False, 
            global_normalize: bool = False,
            norm_low_percentile: Union[float, List[float]] = 0.0,
            norm_high_percentile: Union[float, List[float]] = 100.0
            ):
        if normalize:
            # Ensure percentiles are lists for consistent processing
            if isinstance(norm_low_percentile, (int, float)):
                norm_low_percentile = [norm_low_percentile]
            if isinstance(norm_high_percentile, (int, float)):
                norm_high_percentile = [norm_high_percentile]

            if global_normalize:
                # Use the first percentile value for global normalization
                low_p = norm_low_percentile[0]
                high_p = norm_high_percentile[0]
                img_min = np.percentile(img_data, low_p)
                img_min = np.broadcast_to(img_min, (img_data.shape[2],))  # Broadcast to number of channels
                img_max = np.percentile(img_data, high_p)
                img_max = np.broadcast_to(img_max, (img_data.shape[2],))  # Broadcast to number of channels   
            else:
                # Per-channel normalization
                num_channels = img_data.shape[2]    
                # Broadcast percentile lists if they have a single element
                if len(norm_low_percentile) == 1:
                    norm_low_percentile = norm_low_percentile * num_channels
                if len(norm_high_percentile) == 1:
                    norm_high_percentile = norm_high_percentile * num_channels
                # Calculate percentiles per-channel
                img_reshaped = np.moveaxis(img_data, 2, -1)
                img_min = np.percentile(img_reshaped, norm_low_percentile, axis=(0, 1, 2)) # shape (Channels, Channels)
                img_min = np.diag(img_min) # shape (Channels,)
                img_max = np.percentile(img_reshaped, norm_high_percentile, axis=(0, 1, 2)) # shape (Channels, Channels)
                img_max = np.diag(img_max) # shape (Channels,)
            
            # Avoid division by zero
            range_val = img_max - img_min
            # Use a small epsilon to handle the case where range_val is zero
            range_val[range_val == 0] = np.finfo(float).eps
            # Clip and scale the data
            img_min = img_min[None, None, :, None]  # Reshape for broadcasting
            range_val = range_val[None, None, :, None]  # Reshape for broadcasting
            scaled_data = np.clip((img_data - img_min) / range_val, 0, 1)
            return (scaled_data * 255).astype(np.uint8)

        # if not normalizing then divide by max value of the data type
        if np.issubdtype(img_data.dtype, np.integer):
            return (img_data / np.iinfo(img_data.dtype).max * 255).astype(np.uint8)
        else:
            # For float, assume range is [0, 1] if not normalizing
            return (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)


    @ui_callback
    def redraw_background(self):
        if not self._image_is_loaded:
            return
        self.bg_display, canvas = self._get_rendered_image_display(
            self._normalize_display.value,
            self._normalize_display_global.value,
            str(self._display_normalization_low_percentile.value), #TODO: optimization: only check currently displayed channels?
            str(self._display_normalization_high_percentile.value),
            self.maximum_displayed_channels,
            self.display_channel_offset,
            self.display_layer_offset
        )
        self.canvas[0].draw_image(canvas, 0, 0)

    @lru_cache(maxsize=512)
    def _get_rendered_image_display(
        self, 
        normalize: bool, 
        global_normalize: bool,
        norm_low_percentile: str, # from list to make it hashable
        norm_high_percentile: str, # from list to make it hashable
        maximum_displayed_channels: Literal[1, 2, 3],
        channel_offset: int,
        layer_offset: int
        ):
        bg_display = self.convert_image_data_for_display(
            self.bg_hires, 
            normalize=normalize, 
            global_normalize=global_normalize,
            norm_low_percentile=self.display_normalization_low_percentile, # here we use the original list
            norm_high_percentile=self.display_normalization_high_percentile, # here we use the original list
            maximum_displayed_channels=maximum_displayed_channels,
            channel_offset=channel_offset,
            layer_offset=layer_offset
            )
        canvas = Canvas(width=self.image_size[1], height=self.image_size[0])
        canvas.put_image_data(bg_display, 0, 0)
        return bg_display, canvas


    def get_mask_ids_under_probe(
            self, 
            x: Optional[Union[int, float]] = None, 
            y: Optional[Union[int, float]] = None
            ) -> np.ndarray:
        """
        Get the mask IDs under the probe area.
        If x and y are not provided, use the current probe_x and probe_y.
        Since the probe_x and probe_y are limited by the probe_width and probe_height,
        (so that the probe_x and probe_y are always in the center of the probe area),
        it is possible to provide an unbounded x and y, which will be clamped to the image size.
        The returned mask IDs will be a 2D array of shape (Channels, Layers).
        If the mask is not loaded, raises a ValueError.

        Args:
            x (Optional[Union[int, float]]): The x-coordinate of the probe. Defaults to None.
            y (Optional[Union[int, float]]): The y-coordinate of the probe. Defaults to None.

        Raises:
            ValueError: If the mask is not loaded.

        Returns:
            np.ndarray: The mask IDs under the probe area. A 2D array of shape (Channels, Layers).
        """
        if not self._mask_is_loaded:
            raise ValueError("Mask is not loaded")
        # if no xy is provided, use the current probe xy, otherwise clamp to image size
        if x is None:
            x = self.probe_x
        else:
            x = int(np.round(np.clip(x, 0, self.image_size[1]-1)))
        if y is None:
            y = self.probe_y
        else:
            y = int(np.round(np.clip(y, 0, self.image_size[0]-1)))
        # get the mask IDs under the probe area
        # if same_mask_ids_across_layers and/or same_mask_ids_across_channels is enabled,
        # then only get the mask ID at the channel/layer offsets
        if not self.same_mask_ids_across_layers and not self.same_mask_ids_across_channels:
            mask_ids = self.mask[y, x, :, :]
        elif self.same_mask_ids_across_layers and not self.same_mask_ids_across_channels:
            layer_offset = min(self.display_layer_offset, self.mask.shape[-1] - 1)
            mask_ids = self.mask[y, x, :, layer_offset][..., None]
        elif not self.same_mask_ids_across_layers and self.same_mask_ids_across_channels:
            # for all visible channels (maximum 3) try to get a valid mask ID
            # the first channel with a non-zero mask ID will be used
            for i in range(self.maximum_displayed_channels):
                channel_offset = min(self.display_channel_offset + i, self.mask.shape[-2] - 1)
                mask_ids = self.mask[y, x, channel_offset, :][..., None, :]
                if mask_ids.sum() > 0:
                    break
        else: # self.same_mask_ids_across_layers and self.same_mask_ids_across_channels:
            layer_offset = min(self.display_layer_offset, self.mask.shape[-1] - 1)
            # for all visible channels (maximum 3) try to get a valid mask ID
            # the first channel with a non-zero mask ID will be used
            for i in range(self.maximum_displayed_channels):
                channel_offset = min(self.display_channel_offset + i, self.mask.shape[-2] - 1)
                mask_ids = self.mask[y, x, channel_offset, layer_offset][..., None, None]
                if mask_ids.sum() > 0:
                    break
        return mask_ids


    @ui_callback
    def redraw_mask_display(self):
        if self.nrt:
            return
        if not self._image_is_loaded:
            return
        if not self._mask_is_loaded:
            return
        # if not probing with masks, then clear the mask layer
        if not self.probe_with_mask:
            self.canvas[1].clear()
            return
        self.canvas[1].clear()
        _, canvas = self._get_rendered_mask_display(
            self._get_hashable_mask_ids(self.selected_mask_ids),
            self.unmuted,
            self.maximum_displayed_channels,
            self.display_channel_offset,
            self.display_layer_offset
        )
        if canvas is not None:
            self.canvas[1].draw_image(canvas, 0, 0)


    def _get_hashable_mask_ids(self, arr):
        """Creates a hashable representation of a numpy array, handling object arrays."""
        if arr.dtype == object:
            # For object arrays, create a canonical string representation.
            stringify = np.vectorize(lambda x: str(sorted(x)) if isinstance(x, list) else str(x))
            return ''.join(stringify(arr).flat)
        else:
            # For numeric arrays, use tobytes()
            return arr.tobytes()


    @lru_cache(maxsize=4096)
    def _get_rendered_mask_display(self, mask_ids, unmuted, num_channels, channel_offset, layer_offset):
        mask_filtered, y, x = ids2mask_cropped(self.mask, self.selected_mask_ids, num_channels, channel_offset, layer_offset)
        if mask_filtered is None:
            return None, None
        self._displayed_mask_bbox = (y, x, y+mask_filtered.shape[0], x+mask_filtered.shape[1]) # top, left, bottom, right
        opacity = 0.4
        mask_display = np.repeat(mask_filtered[..., None] * 255, 4, axis=-1)
        mask_display[..., 2] = 0 # turn off blue
        mask_display[..., 3] = opacity * mask_display[..., 0] # set alpha channel
        if unmuted:
            mask_display[..., 1] = 0 # turn off green (becomes red)
        canvas = Canvas(width=self.image_size[1], height=self.image_size[0])
        canvas.put_image_data(mask_display, x, y)
        canvas.stroke_style = 'red' if unmuted else 'yellow'
        canvas.stroke_rect(
            int(x), 
            int(y), 
            *mask_filtered.shape[1::-1]) # width, height
        return mask_display, canvas


    def _compute_mask_arrays(self):
        """Compute and cache the mask arrays for the current selected_mask_ids."""
        if self.selected_mask_ids.ndim == 1:
            selected_mask = np.isin(self.mask, self.selected_mask_ids)
        elif self.selected_mask_ids.ndim == 2:
            if self.selected_mask_ids.dtype != object:
                selected_mask = (self.mask == self.selected_mask_ids)
            else:
                selected_mask = np.zeros_like(self.mask, dtype=bool)
                selected_mask_ids_broadcasted = np.broadcast_to(self.selected_mask_ids, (self._mask_channels, self._mask_layers))
                for chan in range(self._mask_channels):
                    for layer in range(self._mask_layers):
                        ids = selected_mask_ids_broadcasted[chan, layer]
                        if ids:
                            mask_slice = self.mask[:, :, chan, layer]
                            selected_mask[:, :, chan, layer] = np.isin(mask_slice, ids)
        else:
            raise ValueError("selected_mask_ids must be 1D or 2D array")

        unselected_mask = (~selected_mask) | (self.mask == 0)
        unselected_mask_broadcasted = np.broadcast_to(unselected_mask, self.bg_hires.shape)

        # precompute bounding box
        active_2d = np.any(~unselected_mask_broadcasted, axis=(2, 3))
        rows = np.where(active_2d.any(axis=1))[0]
        cols = np.where(active_2d.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            bbox = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
        else:
            bbox = None

        self._cached_mask_result = {
            "unselected_mask_broadcasted": unselected_mask_broadcasted,
            "bbox": bbox
        }


    def get_probe_matrix(self) -> np.ndarray:
        """
        Get the probe matrix from the background image.
        If probing with a mask, it returns the full image as a NumPy MaskedArray,
        with all pixels outside the selected mask ID being masked.
        Otherwise, it returns a rectangular slice of the background image.
        The size of the probe is determined by the probe_width and probe_height properties.
        The position of the probe is determined by the probe_x and probe_y properties.
        The probe is clamped to the image size, so that it doesn't go out of bounds.

        Returns:
            np.ndarray: The probe matrix.
        """
        if self.probe_with_mask and self._mask_is_loaded:
            # compute cached mask if necessary
            if self._cached_mask_result is None:
                self._compute_mask_arrays()

            cached = self._cached_mask_result
            probe = np.ma.masked_array(self.bg_hires, mask=cached["unselected_mask_broadcasted"])
            bbox = cached["bbox"]
            if bbox is not None:
                y_min, y_max, x_min, x_max = bbox
                probe = probe[y_min:y_max, x_min:x_max]

        # if normal rectangular probe        
        else:
            x_from = max(self.probe_x - self.probe_width//2, 0)
            y_from = max(self.probe_y - self.probe_height//2, 0)
            probe = self.bg_hires[y_from : y_from + self.probe_height, x_from : x_from + self.probe_width]
        # apply filter_by_layer_offset and/or filter_by_channel_offset if enabled
        # instead of slicing we are doing this with masking to avoid changing the overall shape shape of the probe
        if self.filter_probe_by_channel_offset or self.filter_probe_by_layer_offset:
            # Start with an empty mask (all False)
            offset_mask = np.zeros(probe.shape, dtype=bool)
            
            # Create a mask for the channel if filter is on
            if self.filter_probe_by_channel_offset and probe.shape[2] > 1:
                channel_mask = np.ones(probe.shape, dtype=bool)
                channel_idx = min(self.display_channel_offset, probe.shape[2] - 1)
                channel_mask[:, :, channel_idx, :] = False
                offset_mask |= channel_mask

            # Create a mask for the layer if filter is on
            if self.filter_probe_by_layer_offset and probe.shape[3] > 1:
                layer_mask = np.ones(probe.shape, dtype=bool)
                layer_idx = min(self.display_layer_offset, probe.shape[3] - 1)
                layer_mask[:, :, :, layer_idx] = False
                offset_mask |= layer_mask

            # If a mask already exists on the probe, combine them
            if np.ma.is_masked(probe):
                offset_mask |= probe.mask

            probe = np.ma.masked_array(probe.data, mask=offset_mask)
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
        prev_state = {key: getattr(self, key) for key in self.TIMELINE_KEYS.keys()}
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
        first_frame_settings = {key: val[0] for key, val in timeline_frames["interpolatable"].items()} # first the interpolatable
        if 0 in timeline_frames["events"]: # then the events
            first_frame_settings = {**first_frame_settings, **timeline_frames["events"][0]} # combine the two
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
            # first fetch the interpolatable keys' settings
            frame_settings = {key: val[frame] for key, val in timeline_frames["interpolatable"].items()}
            if frame in timeline_frames["events"]: # then the events
                frame_settings = {**frame_settings, **timeline_frames["events"][frame]} # combine the two
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
        # restore the state
        for key in self.TIMELINE_KEYS.keys():
            setattr(self, key, prev_state[key])
        AppRegistry().notify_resume(self) # notify other apps to resume their audio
        self.graph.start() # start the global graph
        self._draw_lock = False # disable draw lock
        self.start_compute_thread() # start the compute thread
        self.compute_and_draw() # trigger a draw to update the canvas and update mappings to where they where before the render

        return _output_buffer


    def render_frame(self, frame, settings):
        # set the app to the settings, casting to the expected type
        for key in settings.keys():
            meta = self.TIMELINE_KEYS.get(key, None)
            val = settings[key]
            if meta["type"] is not None:
                val = meta["type"](val)
            setattr(self, key, val)

        # Get probe matrix
        probe_mat = self.get_probe_matrix()

        # Compute probe features
        self.compute_features(probe_mat)

        # Update mappings
        self.compute_mappers(frame=frame)


    def standardize_timeline(self, timeline):
        """Fill in missing values in the timeline with the previous values, and sort the timeline by timepoint."""
        latest_setting = {}
        for key, meta in self.TIMELINE_KEYS.items():
            val = getattr(self, key)
            if meta["type"] is not None:
                val = meta["type"](val)
            latest_setting[key] = val
        sorted_timeline = sorted(timeline, key=lambda x: x[0]) # sort by timepoint
        new_timeline = []
        for timepoint, settings in sorted_timeline:
            new_settings = {**latest_setting, **settings}
            new_timeline.append((timepoint, new_settings))
            latest_setting = new_settings
        return new_timeline
    

    def _separate_interpolatable_from_events(self, timeline):
        """Separate the timeline into interpolatable keys and event keys."""
        interpolatable_keys = [key for key, meta in self.TIMELINE_KEYS.items() if meta["can_interp"]]
        event_keys = [key for key, meta in self.TIMELINE_KEYS.items() if not meta["can_interp"]]
        timeline_interpolatable = []
        timeline_events = []
        for timepoint, settings in timeline:
            interp_settings = {key: val for key, val in settings.items() if key in interpolatable_keys}
            event_settings = {key: val for key, val in settings.items() if key in event_keys}
            timeline_interpolatable.append((timepoint, interp_settings))
            timeline_events.append((timepoint, event_settings))
        return timeline_interpolatable, timeline_events
    

    def _filter_timeline_keys(self, timeline):
        """Filter the timeline to only include keys that are in the timeline keys."""
        filtered_timeline = []
        for timepoint, settings in timeline:
            filtered_settings = {key: val for key, val in settings.items() if key in self.TIMELINE_KEYS.keys()}
            filtered_timeline.append((timepoint, filtered_settings))
        return filtered_timeline


    def generate_timeline_frames(self, timeline, num_frames, fps):
        # initialize the timeline arrays for interpolatable keys
        timeline_frames = {
            "interpolatable": {},
            "events": {}
        }
        for key, meta in self.TIMELINE_KEYS.items():
            if meta["can_interp"]:
                timeline_frames["interpolatable"][key] = np.zeros(num_frames)
            # else:
            #     timeline_frames["events"][key] = {}
        # discard any keys that are not in the timeline keys
        timeline_filtered = self._filter_timeline_keys(timeline)

        # separate interpolatable keys from events
        timeline_interpolatable, timeline_events = self._separate_interpolatable_from_events(timeline_filtered)

        # fill in missing keys for interpolatable keys (by carrying forward the last value) and sort by timepoint
        standardized_timeline = self.standardize_timeline(timeline_interpolatable)

        # fill the timeline arrays for interpolatable keys
        for i in range(len(standardized_timeline) - 1):
            current_time, current_settings = standardized_timeline[i]
            next_time, next_settings = standardized_timeline[i+1]
            current_frame = sec2frame(current_time, fps)
            next_frame = sec2frame(next_time, fps)
            n_frames = next_frame - current_frame

            # for each key, either interpolate between the current and next value
            for key in timeline_frames["interpolatable"].keys():
                current_val = current_settings[key]
                next_val = next_settings[key]
                timeline_frames["interpolatable"][key][current_frame:next_frame] = np.linspace(current_val, next_val, n_frames)

        # fill timeline events
        for timepoint, settings in timeline_events:
            frame = sec2frame(timepoint, fps)
            if settings:  # only add if there are event settings
                if frame not in timeline_frames["events"]:
                    timeline_frames["events"][frame] = {}
                timeline_frames["events"][frame].update(settings)

        return timeline_frames
    

    def compute(self):
        self.compute_event.set()

    def compute_and_draw(self):
        self.compute()
        self.draw()

    @ui_callback
    def draw(self):
        """Render new frames for all kernels, then update the HTML canvas with the results."""
        if self._draw_lock:
            return
        
        # Clear the canvas
        self.canvas[2].clear()

        # Put the probe rectangle to the canvas
        if not self.probe_with_mask:
            self.canvas[2].stroke_style = 'red' if self.unmuted else 'yellow'
            self.canvas[2].stroke_rect(
                int(self.probe_x - self.probe_width//2), 
                int(self.probe_y - self.probe_height//2), 
                int(self.probe_width), 
                int(self.probe_height))
        
        # update the probe_x and probe_y values in the UI
        probe_x_numbox = self._ui_widget_refs.get("probe_x", None)
        if probe_x_numbox is None:
            probe_x_numbox = find_widget_by_tag(self.ui, "probe_x")
            self._ui_widget_refs["probe_x"] = probe_x_numbox
        probe_x_numbox.value = self.probe_x
        probe_y_numbox = self._ui_widget_refs.get("probe_y", None)
        if probe_y_numbox is None:
            probe_y_numbox = find_widget_by_tag(self.ui, "probe_y")
            self._ui_widget_refs["probe_y"] = probe_y_numbox
        probe_y_numbox.value = self.probe_y
        # Flush pending UI (non-blocking, main thread)
        now = time.perf_counter()
        if now - self._last_ui_update_time > self._ui_flush_min_interval:
            self._last_ui_update_time = now
            for obj in chain(getattr(self, "features", []), getattr(self, "synths", [])):
                if getattr(obj, "_pending_ui", False) and hasattr(obj, "update_ui_throttled"):
                    obj.update_ui_throttled(force=False)

        # log probe params and unmuted state
        self._probe_x_on_last_draw = self._probe_x
        self._probe_y_on_last_draw = self._probe_y
        self._probe_width_on_last_draw = self._probe_width.value
        self._probe_height_on_last_draw = self._probe_height.value
        self._unmuted_on_last_draw = self._unmuted


    def mouse_callback(self, x, y, pressed: int = 0):
        """Handle mouse, compute probe features, update synth(s), and render kernels."""
        if self._nrt:
            return # Skip if we are in non-real-time mode
        
        if not self.probe_follows_idle_mouse and pressed < 0 and self.mouse_btn == 0:
            return # Skip if we are not following the idle mouse

        # Drop excess events over the refresh interval
        current_time = time.perf_counter()
        if current_time - self.last_draw_time < self._refresh_interval and pressed < 2: # only skip on mouse move, not on mouse down/up
            return  # Skip if we are processing too quickly
        self.last_draw_time = current_time  # Update the last event time

        with hold_canvas(self.canvas), self.hold_ui():
            # parse mouse event (double-click, click, release)
            if pressed == 2: # Mouse down
                if current_time - self._last_mouse_down_time < 0.2:
                    self.mouse_btn = 2 # Double-click
                else:
                    self.mouse_btn = 1 # Single-click
                self._last_mouse_down_time = current_time
            elif pressed == 3: # Mouse up
                self.mouse_btn = 0
            # If in Select mode, we are either box-selecting the probe or adding/removing mask IDs
            if self.interaction_mode == "Select":
                # upon double-click add/remove mask IDs if probing with mask and mask is loaded
                if self.probe_with_mask and self._mask_is_loaded:
                    if self.mouse_btn == 2:
                        # if selected_mask_ids is not in object mode, we need to convert it to object mode
                        if self.selected_mask_ids.dtype != object:
                            # if 1D, we have to select the same IDs for all channels and layers
                            if self.selected_mask_ids.ndim == 1:
                                current_ids = self.selected_mask_ids.tolist()
                                selected_mask_ids = np.zeros((self._mask_channels, self._mask_layers), dtype=object)
                                for chan in range(self._mask_channels):
                                    for layer in range(self._mask_layers):
                                        selected_mask_ids[chan, layer] = current_ids.copy()
                                self.selected_mask_ids = selected_mask_ids
                            # if 2D we just broadcast it to (channels, layers) and set it to object dtype
                            elif self.selected_mask_ids.ndim == 2:
                                current_ids = self.selected_mask_ids #.astype(np.uint16)
                                selected_mask_ids = np.broadcast_to(current_ids, (self._mask_channels, self._mask_layers))
                                self.selected_mask_ids = selected_mask_ids.astype(object)
                        # get the mask IDs under the probe
                        mask_ids_under_probe = self.get_mask_ids_under_probe(x, y)
                        # if no mask IDs under the probe, do nothing
                        if mask_ids_under_probe.sum() != 0:
                            # broadcast to (channels, layers)
                            mask_ids_under_probe = np.broadcast_to(mask_ids_under_probe, (self._mask_channels, self._mask_layers))
                            # add/remove the mask IDs under the probe to/from the selected_mask_ids
                            for chan in range(self._mask_channels):
                                for layer in range(self._mask_layers):
                                    id = int(mask_ids_under_probe[chan, layer])
                                    if id != 0: # ignore background
                                        current_ids = self.selected_mask_ids[chan, layer]
                                        if isinstance(current_ids, np.ndarray):
                                            current_ids = current_ids.tolist()
                                        elif not isinstance(current_ids, list):
                                            current_ids = [current_ids] if current_ids != 0 else []
                                        if id in current_ids:
                                            current_ids.remove(id)
                                        else:
                                            current_ids.append(id)
                                        self.selected_mask_ids[chan, layer] = current_ids
                            # invalidate the cached mask result since the selected_mask_ids have changed
                            self._cached_mask_result = None
                    # update the mask IDs display
                    self.update_mask_ids_display()
                    # redraw the mask display
                    self.redraw_mask_display()
                    # compute features and mappers
                    self.compute()
                else: # box-select the probe
                    if self.mouse_btn == 0:
                        self._probe_box_select_start_xy = None # reset start xy
                    elif self.mouse_btn == 1:
                        if self._probe_box_select_start_xy is None:
                            self._probe_box_select_start_xy = (x, y)
                        else:
                            x0, y0 = self._probe_box_select_start_xy
                            x1, y1 = x, y
                            # compute new probe params
                            new_width = abs(x1 - x0)
                            new_height = abs(y1 - y0)
                            new_x = int(np.round(np.clip((x0 + x1) / 2, 0, self.image_size[1]-1)))
                            new_y = int(np.round(np.clip((y0 + y1) / 2, 0, self.image_size[0]-1)))
                            # update probe params without triggering a draw
                            self._probe_width.value = max(new_width, 1)
                            self._probe_height.value = max(new_height, 1)
                            self._probe_x = new_x
                            self._probe_y = new_y
                            # draw and compute
                            self.compute_and_draw()

            else: # Hold or Toggle modes
                # Update probe position without triggering a draw
                self._probe_x = self.clamp_probe_x(x)
                self._probe_y = self.clamp_probe_y(y)
                # Update probe features, mappers, and render canvas
                # only draw when any of the probe params or unmuted has changed since the last draw
                if self._probe_changed or self._unmuted_changed:
                    if self.probe_with_mask and self._mask_is_loaded:
                        # when multiple mask IDs on the same channel/layer are selected, then do not update the selected IDs if mouse is within the masks bbox
                        if self._multiple_mask_ids_per_sheet_selected:
                            if not self.mouse_within_mask_bbox(x, y):
                                self.selected_mask_ids = self.get_mask_ids_under_probe(x, y)
                        else: # is single ID per channel/layer, then always update the selected IDs
                            self.selected_mask_ids = self.get_mask_ids_under_probe(x, y)
                    self.compute_and_draw()


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

    @ui_callback
    def toggle_audio_btn(self, value: bool):
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

    @ui_callback
    def update_mask_ids_display(self):
        # object type filters will always be used for selecting multiple IDs on the same channel or layer
        not_object_type = self.selected_mask_ids.dtype != object
        # in case the user provides a single ID, we can also show the numbox
        single_id = self.selected_mask_ids.size == 1
        numbox_condition = (not_object_type and single_id)
        # show either the numbox or the text field
        numbox = self._ui_widget_refs.get("selected_mask_id", None) # get from cache
        if numbox is None:
            numbox = find_widget_by_tag(self.ui, "selected_mask_id")
            self._ui_widget_refs["selected_mask_id"] = numbox # cache it
        numbox.layout.display = "flex" if numbox_condition else "none"
        text = self._ui_widget_refs.get("selected_mask_ids", None) # get from cache
        if text is None:
            text = find_widget_by_tag(self.ui, "selected_mask_ids")
            self._ui_widget_refs["selected_mask_ids"] = text # cache it
        text.layout.display = "none" if numbox_condition else "flex"
        if numbox_condition:
            numbox.value = self.selected_mask_ids[0, 0]
        else:
            text.value = array2str(self.selected_mask_ids, keep_brackets=True)

    @ui_callback
    def update_display_normalization_percentile_display(self, which: Literal['low', 'high']):
        selected_prop_name = f'display_normalization_{which}_percentile'
        not_a_list = not isinstance(getattr(self, selected_prop_name), list)
        single_value = len(getattr(self, selected_prop_name)) == 1 if not not_a_list else False
        slider_condition = not_a_list or single_value
        # show either the slider or the text field
        slider = self._ui_widget_refs.get(f"display_normalization_{which}_percentile_slider", None) # get from cache
        if slider is None:
            slider = find_widget_by_tag(self.ui, f"display_normalization_{which}_percentile_slider")
            self._ui_widget_refs[f"display_normalization_{which}_percentile_slider"] = slider # cache it
        text = self._ui_widget_refs.get(f"display_normalization_{which}_percentile_text", None) # get from cache
        if text is None:
            text = find_widget_by_tag(self.ui, f"display_normalization_{which}_percentile_text")
            self._ui_widget_refs[f"display_normalization_{which}_percentile_text"] = text # cache it
        slider.layout.display = "flex" if slider_condition else "none"
        text.layout.display = "none" if slider_condition else "flex"
        if slider_condition:
            slider.value = getattr(self, selected_prop_name)[0]
        else:
            text.value = array2str(getattr(self, selected_prop_name))


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
