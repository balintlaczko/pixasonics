from ipywidgets import Label, Layout, Box, VBox, GridBox, Button, IntSlider, FloatSlider, ToggleButton, Accordion

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
        self.create_ui()

    def __call__(self):
        return self.card

    def detach_callback(self, b):
        print("Override this method to detach the mapper")

    def create_ui(self):
        mapper_label = Label(
            value="Mapper", 
            style=dict(
                font_weight='bold',
                font_size='2em'))
        
        mapper_id = Label(
            value=self.id, 
            style=dict(
                font_weight='bold',
                font_size='1em',
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
            icon='unlink')
        detach_row = Box(
            [detach_btn], 
            layout=Layout(justify_content='flex-end'))
        detach_btn.on_click(self.detach_callback)

        self.card = GridBox(
            children=[top_row, from_row, to_row, detach_row],
            layout=Layout(
                width='auto', 
                grid_template_columns='auto', 
                grid_template_rows='2fr 1.5fr 1.5fr 1fr',
                max_width='300px',
                min_height='200px',
                border='2px solid black',
                padding='5px',
                margin='5px'))
        

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