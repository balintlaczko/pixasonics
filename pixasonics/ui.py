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
    
    def w_slider_callback(self, change):
        """override this method to handle width slider changes"""
        pass

    def h_slider_callback(self, change):
        """override this method to handle height slider changes"""
        pass

    def create_ui(self):
        probe_w_label = Label(value="Probe Width:")
        probe_w_slider = IntSlider(value=50, min=1, max=500, step=1)
        probe_w_box = Box(
            [probe_w_label, probe_w_slider], 
            layout=Layout(
                justify_content='space-around', 
                align_items='flex-start', 
                flex_flow='column',
                padding='5px'))

        probe_h_label = Label(value="Probe Height:")
        probe_h_slider = IntSlider(value=50, min=1, max=500, step=1)
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
        
        probe_w_slider.observe(self.w_slider_callback, names='value')
        probe_h_slider.observe(self.h_slider_callback, names='value')


class AudioSettings():
    def __init__(self):
        self.create_ui()

    def __call__(self):
        return self.box
    
    def switch_callback(self, change):
        """override this method to handle switch changes"""
        pass

    def volume_slider_callback(self, change):
        """override this method to handle volume slider changes"""
        pass

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

        master_volume_label = Label(value="Master Volume (dB):")

        master_volume_slider = FloatSlider(
            value=0,
            min=-36,
            max=0,
            step=0.01,
            orientation='horizontal',
            layout=Layout(width='100%', height='auto')
        )

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
        
        audio_switch.observe(self.switch_callback, names='value')
        master_volume_slider.observe(self.volume_slider_callback, names='value')


class AppUI():
    def __init__(
            self,
            probe_settings, 
            audio_settings,
            canvas_width=500,
            canvas_height=500, 
            ):
        self.probe_settings = probe_settings
        self.audio_settings = audio_settings
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        self.create_ui()

    def __call__(self):
        return self.box

    def create_ui(self):
        features_carousel = VBox([])
        synths_carousel = VBox([])
        mappers_carousel = VBox([])

        app_canvas = Box(
            [],
            layout=Layout(
                width=f'{self.canvas_width}px', 
                height=f'{self.canvas_height}px')
        )

        app_settings = Accordion(
            children=[
                self.probe_settings(), 
                self.audio_settings(),
                features_carousel, 
                synths_carousel, 
                mappers_carousel],
            titles=('Probe', 'Audio', "Features", "Synths", "Mappers"),
            layout=Layout(width='400px', min_width='300px', max_width='400px'))

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
                justify_content='space-around'))
 