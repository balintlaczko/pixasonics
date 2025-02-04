from ipywidgets import Label, Layout, Box, VBox, GridBox, Button, IntSlider, FloatSlider, FloatLogSlider, ToggleButton, Accordion, Text
from math import log10
class MapperCard():
    def __init__(
            self, 
            id: str = "# ID", 
            from_name: str = "(mapping source)",
            to_name: str = "(mapping target)",
    ):
        self.id = id
        self.from_name = from_name
        self.to_name = to_name
        self.app = None
        self.mapper = None
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        print("MapperCard: detaching mapper", self.id)
        if self.app is not None and self.mapper is not None:
            self.app.detach_mapper(self.mapper)

    def create_ui(self):
        mapper_label = Label(
            value="Mapper", 
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
            layout=Layout(max_width='80px'))
        detach_row = Box(
            [detach_btn], 
            layout=Layout(justify_content='flex-end'))
        detach_btn.on_click(self.detach_callback)

        self.card = GridBox(
            children=[top_row, from_row, to_row, detach_row],
            layout=Layout(
                width='auto', 
                grid_template_columns='auto', 
                grid_template_rows='1fr 0.8fr 1.6fr 1fr',
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
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        print("FeatureCard: detaching feature", self.id)
        if self.app is not None and self.feature is not None:
            self.app.detach_feature(self.feature)

    def reset_callback(self, b):
        print("FeatureCard: resetting min max", self.id)
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
            layout=Layout(max_width='80px'))
        reset_btn.on_click(self.reset_callback)

        detach_btn = Button(
            description="Detach", 
            button_style='danger', 
            icon='unlink',
            layout=Layout(max_width='80px'))
        detach_btn.on_click(self.detach_callback)
        
        btn_row = Box(
            [reset_btn, detach_btn], 
            layout=Layout(
                width='100%',
                justify_content='space-between'))

        self.card = Box(
            children=[top_block, min_block, max_block, value_block, btn_row],
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
    ):
        self.name = name
        self.id = id
        self.params = params
        self.app = None
        self.synth = None
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        print("SynthCard: detaching synth", self.id)
        if self.app is not None and self.synth is not None:
            self.app.detach_synth(self.synth)

    def reset_callback(self, b):
        print("SynthCard: resetting to default params", self.id)
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
            # if the param has the 'scale' key, use it to scale the slider
            if param['scale'] == 'log':
                param_slider = FloatLogSlider(
                    value=param['default'],
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
                    value=param['default'],
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
                lambda change: self.synth.set_input_buf(
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
            layout=Layout(max_width='80px'))
        reset_btn.on_click(self.reset_callback)

        detach_btn = Button(
            description="Detach", 
            button_style='danger', 
            icon='unlink',
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
        

class ProbeSettings():
    def __init__(self):
        self.create_ui()

    def __call__(self):
        return self.box

    def create_ui(self):
        probe_w_label = Label(value="Probe Width:")
        probe_w_slider = IntSlider(value=50, min=1, max=500, step=1)
        probe_w_slider.tag = "probe_w_slider"
        probe_w_box = Box(
            [probe_w_label, probe_w_slider], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))

        probe_h_label = Label(value="Probe Height:")
        probe_h_slider = IntSlider(value=50, min=1, max=500, step=1)
        probe_h_slider.tag = "probe_h_slider"
        probe_h_box = Box(
            [probe_h_label, probe_h_slider], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))

        self.box = Box(
            [probe_w_box, probe_h_box], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                # border='2px solid black',
                # padding='5px',
                # margin='5px'
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
                max_width='100px',
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
            layout=Layout(width='100%', height='auto')
        )
        master_volume_slider.tag = "master_volume_slider"

        master_volume_box = Box(
            [master_volume_label, master_volume_slider],
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))

        self.box = GridBox(
            children=[audio_switch, master_volume_box],
            layout=Layout(
                width='auto', 
                grid_template_columns='auto auto', 
                grid_template_rows='auto',
                # border='2px solid black',
                # padding='5px',
                # margin='5px'
                ))


class AppUI():
    def __init__(
            self,
            probe_settings, 
            audio_settings,
            canvas_width=500,
            canvas_height=500, 
            ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        self.create_ui(probe_settings, audio_settings)

    def __call__(self):
        return self.box

    def create_ui(self, probe_settings, audio_settings):
        features_carousel = VBox([])
        features_carousel.tag = "features_carousel"
        synths_carousel = VBox([])
        synths_carousel.tag = "synths_carousel"
        mappers_carousel = VBox([])
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
                probe_settings(), 
                audio_settings(),
                features_carousel, 
                synths_carousel, 
                mappers_carousel],
            titles=('Probe', 'Audio', "Features", "Synths", "Mappers"),
            layout=Layout(width='400px', min_width='300px', max_width='400px'))
        app_settings.tag = "app_settings"

        app_settings_container = Box(
            [app_settings], 
            layout=Layout(
                overflow='scroll',
                # border='3px solid black',
                padding='5px',
                max_height='500px',))

        self.box = Box(
            [app_canvas, app_settings_container], 
            layout=Layout(
                width='auto', 
                height='auto', 
                justify_content='center'))


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
            if change['name'] == 'value':
                self.value = change['new']
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