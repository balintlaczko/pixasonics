from ipycanvas import Canvas, hold_canvas
from ipywidgets import Label, Layout, Box, VBox, HBox, GridBox, Button, IntSlider, FloatSlider, FloatLogSlider, ToggleButton, Accordion, Text, FloatText, IntText, BoundedFloatText, ToggleButtons, Checkbox
from math import log10
from .utils import array2str, scale_array_exp
import numpy as np

class MapperCard():
    def __init__(
            self, 
            name: str = "Mapper",
            id: str = "# ID", 
            from_name: str = "(mapping source)",
            to_name: str = "(mapping target)",
    ):
        self.name = name
        self.id = id
        self.from_name = from_name
        self.to_name = to_name
        self.app = None
        self.mapper = None
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        if self.app is not None and self.mapper is not None:
            self.app.detach_mapper(self.mapper)

    def create_ui(self):
        mapper_label = Label(
            value=self.name, 
            style=dict(
                font_weight='bold',
                font_size='20px'))
        
        mapper_id = Label(
            value="#" + self.id, 
            style=dict(
                font_weight='bold',
                font_size='10px',
                text_color='gray'))

        top_row = Box(
            [mapper_label, mapper_id], 
            layout=Layout(justify_content='space-between'))

        from_label = Label(value="From:")
        from_value = Label(value=self.from_name)
        from_row = Box(
            [from_label, from_value], 
            layout=Layout(justify_content='space-between'))

        to_label = Label(value="To:")
        to_value = Label(value=self.to_name)
        to_row = Box(
            [to_label, to_value], 
            layout=Layout(justify_content='space-between'))

        detach_btn = Button(
            description="Detach", 
            button_style='danger', 
            icon='unlink',
            tooltip='Detach from the app',
            layout=Layout(max_width='80px'))
        detach_row = Box(
            [detach_btn], 
            layout=Layout(justify_content='flex-end'))
        detach_btn.on_click(self.detach_callback)

        self.card = VBox(
            children=[top_row, from_row, to_row, detach_row],
            layout=Layout(
                width='auto',
                justify_content='flex-start',
                max_width='260px',
                min_height='140px',
                border='1px solid black',
                padding='5px',
                margin='5px'))
        self.card.tag = f"mapper_{self.id}"


class FeatureCard():
    def __init__(
            self,
            name: str = "Feature", 
            id: str = "# ID", 
            min: str = "(str(min array))",
            max: str = "(str(max array))",
            value: str = "(str(value array))",
    ):
        self.name = name
        self.id = id
        self.min = min
        self.max = max
        self.value = value
        self.app = None
        self.feature = None
        self.num_features = 1 # default to 1 feature
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        if self.app is not None and self.feature is not None:
            self.app.detach_feature(self.feature)

    def reset_callback(self, b):
        if self.feature is not None:
            self.feature.reset_minmax()

    def create_ui(self):
        feature_label = Label(
            value=self.name, 
            style=dict(
                font_weight='bold',
                font_size='20px'))
        
        feature_id = Label(
            value="#" + self.id, 
            style=dict(
                font_weight='bold',
                font_size='10px',
                text_color='gray'))

        top_block = Box(
            [feature_label, feature_id], 
            layout=Layout(
                justify_content='space-between',
                align_items='flex-start',
                flex_flow='row',
                width='100%'))
        
        num_features_label = Label(value="Number of Features:")
        num_features_value = IntText(
            value=self.num_features,
            disabled=True,
            layout=Layout(width='60px')
        )
        num_features_value.tag = "num_features"

        num_features_block = Box(
            [num_features_label, num_features_value],
            layout=Layout(
                justify_content='space-between',
                flex_flow='row',
                width='100%'))

        min_label = Label(value="Min:")
        min_value = Text(
            value=self.min,
            placeholder='(min array)',
            description='',
            disabled=True,
            layout=Layout(width='80%')
        )
        min_value.tag = "min"
        min_block = Box(
            [min_label, min_value], 
            layout=Layout(
                justify_content='space-between',
                flex_flow='row',
                width='100%'))
        
        max_label = Label(value="Max:")
        max_value = Text(
            value=self.max,
            placeholder='(max array)',
            description='',
            disabled=True,
            layout=Layout(width='80%')
        )
        max_value.tag = f"max"
        max_block = Box(
            [max_label, max_value], 
            layout=Layout(
                justify_content='space-between',
                flex_flow='row',
                width='100%'))
        
        value_label = Label(value="Value:")
        value_value = Text(
            value=self.value,
            placeholder='(value array)',
            description='',
            disabled=True,
            layout=Layout(width='80%')
        )
        value_value.tag = f"value"
        value_block = Box(
            [value_label, value_value], 
            layout=Layout(
                justify_content='space-between',
                flex_flow='row',
                width='100%'))


        reset_btn = Button(
            description="Reset", 
            button_style='warning', 
            icon='refresh',
            tooltip='Reset to the running min and max',
            layout=Layout(max_width='80px'))
        reset_btn.on_click(self.reset_callback)

        detach_btn = Button(
            description="Detach", 
            button_style='danger', 
            icon='unlink',
            tooltip='Detach from the app',
            layout=Layout(max_width='80px'))
        detach_btn.on_click(self.detach_callback)
        
        btn_row = Box(
            [reset_btn, detach_btn], 
            layout=Layout(
                width='100%',
                justify_content='space-between'))

        self.card = Box(
            children=[top_block, num_features_block, min_block, max_block, value_block, btn_row],
            layout=Layout(
                width='auto', 
                flex_flow='column',
                align_items='flex-start',
                justify_content='flex-start',
                max_width='260px',
                min_height='100px',
                border='1px solid black',
                padding='5px',
                margin='5px'))
        self.card.tag = f"feature_{self.id}"


class SynthCard():
    def __init__(
            self,
            name: str = "Synth", 
            id: str = "# ID",
            params: dict = {},
            num_channels: int = 1,
    ):
        self.name = name
        self.id = id
        self.params = params
        self.num_channels = num_channels
        self.app = None
        self.synth = None
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        if self.app is not None and self.synth is not None:
            self.app.detach_synth(self.synth)

    def reset_callback(self, b):
        if self.synth is not None:
            self.synth.reset_to_default()

    def create_ui(self):        
        synth_label = Label(
            value=self.name, 
            style=dict(
                font_weight='bold',
                font_size='20px'))
        
        synth_id = Label(
            value="#" + self.id, 
            style=dict(
                font_weight='bold',
                font_size='10px',
                text_color='gray'))

        top_block = Box(
            [synth_label, synth_id], 
            layout=Layout(
                justify_content='space-between',
                align_items='flex-start',
                flex_flow='row',
                width='100%'))
        
        # create a block with a float slider for each parameter
        param_blocks = []
        for param_name, param in self.params.items():
            label_str = f"{param_name} ({param['unit']})" if len(param['unit']) > 0 else param_name
            param_label = Label(value=label_str)
            # if the corresponding synth is single channel, use sliders for params, otherwise use text boxes
            if self.num_channels > 1:
                param_slider = Text(
                    value=array2str(param['default']),
                    placeholder='(default)',
                    description='',
                    disabled=True,
                    layout=Layout(width='65%'))
                param_slider.tag = param_name
            else:
                # if the param has the 'scale' key, use it to scale the slider
                if param['scale'] == 'log':
                    param_slider = FloatLogSlider(
                        value=param['default'][0],
                        base=10,
                        min=log10(param['min']),
                        max=log10(param['max']),
                        step=0.0001,
                        description="",
                        readout_format='.1f',
                        layout=Layout(width='65%')
                    )
                elif param['scale'] == 'linear':
                    param_slider = FloatSlider(
                        value=param['default'][0],
                        min=param['min'],
                        max=param['max'],
                        step=0.01,
                        description="",
                        layout=Layout(width='65%')
                    )
                else:
                    raise ValueError(f"SynthCard: Unknown scale '{param['scale']}' for parameter '{param_name}'")
                param_slider.tag = param_name
                param_slider.observe(
                    lambda change: self.synth.set_input(
                        change["owner"].tag, 
                        change["new"],
                        from_slider=True
                    ), 
                    names="value")
            param_block = Box(
                [param_label, param_slider], 
                layout=Layout(
                    justify_content='space-between',
                    flex_flow='row',
                    width='100%'))
            param_blocks.append(param_block)

        reset_btn = Button(
            description="Reset", 
            button_style='warning', 
            icon='refresh',
            tooltip='Reset parameters to their default values',
            layout=Layout(max_width='80px'))
        reset_btn.on_click(self.reset_callback)

        detach_btn = Button(
            description="Detach", 
            button_style='danger', 
            icon='unlink',
            tooltip='Detach from the app',
            layout=Layout(max_width='80px'))
        detach_btn.on_click(self.detach_callback)
        
        btn_row = Box(
            [reset_btn, detach_btn], 
            layout=Layout(
                width='100%',
                justify_content='space-between'))
        
        all_children = [top_block] + param_blocks + [btn_row]

        self.card = Box(
            children=all_children,
            layout=Layout(
                width='auto', 
                flex_flow='column',
                align_items='flex-start',
                justify_content='flex-start',
                max_width='320px',
                min_height='100px',
                border='1px solid black',
                padding='5px',
                margin='5px'))
        self.card.tag = f"synth_{self.id}"


class ADSRCanvas():
    def __init__(self, width=200, height=100, parent_ui=None):
        self.width = width
        self.height = height
        self.parent_ui = parent_ui
        self.canvas = Canvas(width=width, height=height)
        self.draw()

    def __call__(self):
        return self.canvas

    @property
    def attack(self):
        return self.parent_ui.attack if self.parent_ui is not None else 1

    @property
    def decay(self):
        return self.parent_ui.decay if self.parent_ui is not None else 1

    @property
    def sustain(self):
        return self.parent_ui.sustain if self.parent_ui is not None else 1

    @property
    def release(self):
        return self.parent_ui.release if self.parent_ui is not None else 1

    def draw(self):
        total_length = self.attack + self.decay + self.release
        length_coeff = self.width / total_length
        with hold_canvas(self.canvas):
            self.canvas.clear()
            self.canvas.stroke_style = "black"
            self.canvas.line_width = 2
            self.canvas.begin_path()
            # move to the bottom left corner
            self.canvas.move_to(0, self.height)
            # draw the attack
            self.canvas.line_to(self.attack * length_coeff, 0)
            # draw the decay
            self.canvas.line_to((self.attack + self.decay) * length_coeff, (1 - self.sustain) * self.height)
            # draw the release
            self.canvas.line_to(self.width, self.height)
            self.canvas.stroke()
            # put a red circle at each break-point
            self.canvas.stroke_style = "red"
            # at attack point
            self.canvas.stroke_circle(0, self.height, 3)
            # at decay point
            self.canvas.stroke_circle(self.attack * length_coeff, 0, 3)
            # at sustain point
            self.canvas.stroke_circle((self.attack + self.decay) * length_coeff, (1 - self.sustain) * self.height, 3)
            # at release point
            self.canvas.stroke_circle(self.width, self.height, 3)


class EnvelopeCard():
    def __init__(
            self,
            name: str = "Envelope", 
            id: str = "# ID",
            params: dict = {},
    ):
        self.name = name
        self.id = id
        self.params = params
        self.app = None
        self.envelope = None

        # private attributes
        self._attack = Model(params["attack"]["default"])
        self._decay = Model(params["decay"]["default"])
        self._sustain = Model(params["sustain"]["default"])
        self._release = Model(params["release"]["default"])

        self.create_ui()
        self.update()

    @property
    def attack(self):
        return self._attack.value
    
    @attack.setter
    def attack(self, value):
        self._attack.value = value
        self.update()

    @property
    def decay(self):
        return self._decay.value
    
    @decay.setter
    def decay(self, value):
        self._decay.value = value
        self.update()

    @property
    def sustain(self):
        return self._sustain.value

    @sustain.setter
    def sustain(self, value):
        self._sustain.value = value
        self.update()

    @property
    def release(self):
        return self._release.value
    
    @release.setter
    def release(self, value):
        self._release.value = value
        self.update()

    def update_duration(self):
        duration = find_widget_by_tag(self.card, "duration")
        duration.value = round(self.attack + self.decay + self.release, 4)

    def update(self):
        if self.envelope is not None:
            self.envelope.set_param_from_ui("attack", self.attack)
            self.envelope.set_param_from_ui("decay", self.decay)
            self.envelope.set_param_from_ui("sustain", self.sustain)
            self.envelope.set_param_from_ui("release", self.release)
        self.update_duration()
        self.canvas.draw()

    def __call__(self):
        return self.card
    
    def create_ui(self):
        self.create_card()
        self.bind_models()

    def bind_models(self):
        attack = find_widget_by_tag(self.card, "attack")
        self._attack.bind_widget(attack, extra_callback=self.update)

        decay = find_widget_by_tag(self.card, "decay")
        self._decay.bind_widget(decay, extra_callback=self.update)

        sustain = find_widget_by_tag(self.card, "sustain")
        self._sustain.bind_widget(sustain, extra_callback=self.update)

        release = find_widget_by_tag(self.card, "release")
        self._release.bind_widget(release, extra_callback=self.update)

    
    def create_card(self):
        envelope_label = Label(
            value=self.name, 
            style=dict(
                font_weight='bold',
                font_size='14px'))

        envelope_id = Label(
            value="#" + self.id, 
            style=dict(
                font_weight='bold',
                font_size='10px',
                text_color='gray'))

        top_block = Box(
            [envelope_label, envelope_id], 
            layout=Layout(
                justify_content='space-between'))

        param_boxes = []
        for param, param_dict in self.params.items():
            value = param_dict["default"]
            param_label = Label(value=param.capitalize())
            param_numbox = BoundedFloatText(
                value=value,
                step=param_dict["step"],
                min=param_dict["min"],
                max=param_dict["max"],
                layout=Layout(width='90%'),
)
            param_numbox.tag = param
            param_box = VBox(
                [param_label, param_numbox], 
                layout=Layout(
                    justify_content='space-between',
                    height='100%'
                    )
            )
            param_boxes.append(param_box)
        params_display = HBox(
            param_boxes,
            layout=Layout(
                justify_content='space-between',
                width='100%'
            )
        )
        self.canvas = ADSRCanvas(500, 100, self)
        self.canvas.tag = "canvas"
        canvas_wrapper = VBox(
            [self.canvas()],
            layout=Layout(
                justify_content='center',
                width='100%',
                padding='4px'
            )
        )

        duration_label = Label(value="Duration (s):")
        duration_value = FloatText(
            value=0,
            disabled=True,
            layout=Layout(width='auto'))
        duration_value.tag = "duration"
        duration_box = HBox(
            [duration_label, duration_value], 
            layout=Layout(
                justify_content='flex-start',
                width='100%'
            )
        )

        self.card = VBox(
            [top_block, params_display, canvas_wrapper, duration_box],
            layout=Layout(
                justify_content='space-between',
                border='1px solid black',
                min_width='280px',
                max_width='500px',
                width='97%',
                min_height='210px',
                max_height='400px',
                height='210px',
                margin='5px',
                padding='5px'
            )
        )
        

class ProbeSettings():
    def __init__(self, canvas_width=500, canvas_height=500):
        self.create_ui(canvas_width, canvas_height)

    def __call__(self):
        return self.box

    def create_ui(self, canvas_width, canvas_height):
        probe_w_label = Label(value="Probe Width:")
        probe_w_slider = IntSlider(
            value=50, 
            min=1, 
            max=canvas_width, 
            step=1,
            layout=Layout(width='70%')
            )
        probe_w_slider.tag = "probe_w_slider"
        probe_w_box = Box(
            [probe_w_label, probe_w_slider], 
            layout=Layout(
                width='100%',
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='row',
                padding='5px'))

        probe_h_label = Label(value="Probe Height:")
        probe_h_slider = IntSlider(
            value=50, 
            min=1, 
            max=canvas_height, 
            step=1,
            layout=Layout(width='70%')
            )
        probe_h_slider.tag = "probe_h_slider"
        probe_h_box = Box(
            [probe_h_label, probe_h_slider], 
            layout=Layout(
                width='100%',
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='row',
                padding='5px'))
        
        interaction_mode_label = Label(value="Interaction Mode:")
        interaction_mode_buttons = ToggleButtons(
            options=['Hold', 'Toggle'],
            value='Hold',
            button_style='',
            tooltips=['Sound while mouse down', 'Double-click to start sound, double-click again to stop'],
            layout=Layout(padding='0px 0px 0px 10px'),
            style=dict(
                button_width='70px')
        )
        interaction_mode_buttons.tag = "interaction_mode_buttons"
        interaction_mode_box = Box(
            [interaction_mode_label, interaction_mode_buttons],
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='row',
                padding='5px'))
        
        probe_follows_idle_mouse_checkbox = Checkbox(
            value=False,
            description='Probe follows idle mouse',
            tooltip='When enabled, the probe will follow the mouse when not interacting with the canvas',
            indent=False,
            layout=Layout(
                width='auto',
                height='auto')
        )
        probe_follows_idle_mouse_checkbox.tag = "probe_follows_idle_mouse"
        
        probe_x_label = Label(value="Probe X:")
        probe_x_value = IntText(
            value=0, 
            disabled=True,
            layout=Layout(width='90%')
            )
        probe_x_value.tag = "probe_x"
        probe_x_box = VBox(
            [probe_x_label, probe_x_value],
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))
        
        probe_y_label = Label(value="Probe Y:")
        probe_y_value = IntText(
            value=0, 
            disabled=True,
            layout=Layout(width='90%')
            )
        probe_y_value.tag = "probe_y"
        probe_y_box = VBox(
            [probe_y_label, probe_y_value],
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))
        
        probe_xy_box = HBox(
            [probe_x_box, probe_y_box],
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='row'))

        self.box = Box(
            [probe_w_box, probe_h_box, interaction_mode_box, probe_follows_idle_mouse_checkbox, probe_xy_box], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                ))


class AudioSettings():
    def __init__(self):
        self.create_ui()

    def __call__(self):
        return self.box

    def create_ui(self):
        audio_switch = ToggleButton(
            value=False,
            description='Audio',
            tooltip='Enable/disable audio processing',
            icon='volume-up',
            layout=Layout(
                width='auto', 
                max_width='90px',
                height='auto')
        )
        audio_switch.tag = "audio_switch"

        master_volume_label = Label(value="Master Volume (dB):")

        master_volume_slider = FloatSlider(
            value=0,
            min=-36,
            max=0,
            step=0.01,
            orientation='horizontal',
            layout=Layout(width='90%', height='auto')
        )
        master_volume_slider.tag = "master_volume_slider"

        master_volume_box = Box(
            [master_volume_label, master_volume_slider],
            layout=Layout(
                width='70%',
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))

        self.master_box = Box(
            children=[audio_switch, master_volume_box],
            layout=Layout(
                width='100%', 
                justify_content='space-between',
                max_height='80px'
                ))
        
        recording_path_label = Label(value="Recording Path:")
        recording_path = Text(
            value="recording.wav",
            placeholder='recording.wav',
            description='',
            layout=Layout(width='90%')
        )
        recording_path.tag = "recording_path"
        recording_path_box = VBox(
            [recording_path_label, recording_path],
            layout=Layout(
                width='70%',
                justify_content='space-around', 
                align_items='flex-start', 
                padding='5px'))
        recording_toggle = ToggleButton(
            value=False,
            description='Record',
            tooltip='Start/stop real-time audio recording to the specified file',
            icon='microphone',
            layout=Layout(
                width='auto', 
                max_width='90px',
                height='auto')
        )
        recording_toggle.tag = "recording_toggle"
        recording_box = HBox(
            [recording_path_box, recording_toggle],
            layout=Layout(
                width='100%',
                max_height='80px',
                justify_content='space-between'))
        
        self.box = VBox(
            [self.master_box, recording_box], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                ))
        self.box.tag = "audio_settings"


class ImageSettings():
    def __init__(self):
        self.create_ui()

    def __call__(self):
        return self.box

    def create_ui(self):
        normalize_display = Checkbox(
            value=False,
            description='Normalize display',
            tooltip='Normalize the image displayed on the canvas',
            indent=False,
            layout=Layout(
                width='auto',
                height='auto')
        )
        normalize_display.tag = "normalize_display"

        normalize_display_global = Checkbox(
            value=False,
            description='Global normalization',
            tooltip='Use global min and max for normalization (otherwise use channel-wise min and max)',
            indent=False,
            layout=Layout(
                width='auto',
                height='auto')
        )
        normalize_display_global.tag = "normalize_display_global"

        normalize_box = HBox(
            [normalize_display, normalize_display_global],
            layout=Layout(
                width='90%',
                justify_content='space-around', 
                align_items='flex-start', 
                padding='5px'))
        
        channel_offset = IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            disabled=True,
            description='Channel Offset:',
            orientation='horizontal',
            layout=Layout(width='90%'),
            style=dict(
                description_width='100px')
        )
        channel_offset.tag = "channel_offset"

        layer_offset = IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            disabled=True,
            description='Layer Offset:',
            orientation='horizontal',
            layout=Layout(width='90%'),
            style=dict(
                description_width='100px')
        )
        layer_offset.tag = "layer_offset"
        
        self.box = VBox(
            [normalize_box, channel_offset, layer_offset], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                ))
        self.box.tag = "image_settings"


class AppUI():
    def __init__(
            self,
            audio_settings,
            image_settings,
            probe_settings, 
            canvas_width=500,
            canvas_height=500, 
            ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        self.create_ui(audio_settings, image_settings, probe_settings, canvas_height)

    def __call__(self):
        return self.box

    def create_ui(self, audio_settings, image_settings, probe_settings, canvas_height):
        features_carousel = VBox([], layout=Layout(overflow='scroll'))
        features_carousel.tag = "features_carousel"
        synths_carousel = VBox([], layout=Layout(overflow='scroll'))
        synths_carousel.tag = "synths_carousel"
        mappers_carousel = VBox([], layout=Layout(overflow='scroll'))
        mappers_carousel.tag = "mappers_carousel"

        app_canvas = Box(
            [],
            layout=Layout(
                width=f'{self.canvas_width}px',
                min_width=f'{self.canvas_width}px', 
                height=f'{self.canvas_height}px',
                min_height=f'{self.canvas_height}px',
                border='1px solid black',
                margin='5px',)
        )
        app_canvas.tag = "app_canvas"

        app_settings = Accordion(
            children=[
                audio_settings(),
                image_settings(),
                probe_settings(), 
                features_carousel, 
                synths_carousel, 
                mappers_carousel],
            titles=('Audio Settings', 'Display Settings', 'Probe Settings', "Features", "Synths", "Mappers"),
            layout=Layout(width='400px', min_width='300px', max_width='400px'))
        app_settings.tag = "app_settings"

        app_settings_container = Box(
            [app_settings], 
            layout=Layout(
                overflow='auto',
                padding='5px',
                max_height=f'{canvas_height}px',))

        self.box = Box(
            [app_canvas, app_settings_container], 
            layout=Layout(
                width='auto', 
                height='auto', 
                justify_content='center'))
        

class ExponentCanvas():
    def __init__(self, width=200, height=200, exponent=1):
        self.width = width
        self.height = height
        self._exponent = exponent
        self.canvas = Canvas(width=width, height=height)
        self.draw()

    def __call__(self):
        return self.canvas
    
    @property
    def exponent(self):
        return self._exponent
    
    @exponent.setter
    def exponent(self, value):
        self._exponent = value
        self.draw()

    def draw(self):
        with hold_canvas(self.canvas):
            self.canvas.clear()
            x = np.linspace(0, 1, self.width)
            y = scale_array_exp(x, 0, 1, 0, 1, self._exponent)
            y = 1 - y
            y = y * self.height
            self.canvas.fill_style = "black"
            self.canvas.fill_rects(x * self.width, y, 1, self.height)


class ExponentPlot():
    def __init__(self, width=1000, height=200, exponent=1):
        self.width = width
        self.height = height
        self._exponent = exponent
        
        self.create_ui()

    def __call__(self):
        return self.card
    
    @property
    def exponent(self):
        return self._exponent
    
    @exponent.setter
    def exponent(self, value):
        self._exponent = value
        self.card.children[1].value = value
    
    def create_ui(self):
        canvas = ExponentCanvas(self.width, self.height, self.exponent)
        exp_slider = FloatLogSlider(
            value=self.exponent,
            base=10,
            min=log10(0.01),
            max=log10(100),
            step=0.00001,
            description='Exponent:',
            continuous_update=True,
            readout_format='.4f',
            layout=Layout(padding='20px 0px 10px 0px')
        )
        exp_slider.observe(lambda change: setattr(canvas, "exponent", change.new), names="value")
        self.card = VBox(
            [canvas(), exp_slider],
            layout=Layout(
                padding='10px',
                justify_content='center',
                align_items='center',
                width='auto',)
        )


class Model():
    def __init__(self, val = None):
        self._value = val
        self._widget = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = new_val
        # If a widget is linked, update its value
        if self._widget and self._widget.value != new_val:
            self._widget.value = new_val

    def bind_widget(self, widget, extra_callback=None):
        # Store the widget reference
        self._widget = widget
        
        # Update the class attribute when the widget changes
        def on_widget_change(change, extra_callback):
            if change['name'] == 'value' and change['new'] != self._value:
                self._value = change['new']
                if extra_callback is not None:
                    extra_callback()
        
        widget.observe(lambda x : on_widget_change(x, extra_callback=extra_callback) , names='value')


# Function to search recursively by tag
def find_widget_by_tag(container, tag):
    """
    Recursively search through a container for a widget with a specific custom tag.

    Args:
        container: A widget container (e.g., VBox, HBox).
        tag: The custom tag to search for.

    Returns:
        The widget if found, otherwise None.
    """
    # Check if the container itself has the tag
    if hasattr(container, 'tag') and container.tag == tag:
        return container

    # If the container has children, search recursively
    if hasattr(container, 'children'):
        for child in container.children:
            found_widget = find_widget_by_tag(child, tag)
            if found_widget:
                return found_widget