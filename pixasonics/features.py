import numpy as np
from ipywidgets import widgets
from IPython.display import display
import signalflow as sf

class Feature():
    """Feature base class"""
    def __init__(self, num_features=1, name="Feature"):
        self.num_features = num_features
        self.name = name
        self.features = sf.Buffer(1, num_features)
        self.min = np.zeros_like(self.features.data) * 1e6
        self.max = np.zeros_like(self.features.data) * -1e6
        self.create_widget()

    def __call__(self, mat):
        self.features.data[:, :] = self.compute(mat)
        self.update()
    
    def compute(self, mat):
        return NotImplemented

    def create_widget(self):
        # create a widget with two text boxes for min and max
        # create a box where all the text boxes go
        self.widgets = []
        for i in range(self.num_features):
            min_text = widgets.Text(
                value='',
                description=f'{self.name}_{i}_Min:',
                disabled=True,
                style=dict(description_width='auto')
            )
            max_text = widgets.Text(
                value='',
                description=f'{self.name}_{i}_Max:',
                disabled=True,
                style=dict(description_width='auto')
            )
            last_text = widgets.Text(
                value='',
                description=f'{self.name}_{i}_Last:',
                disabled=True,
                style=dict(description_width='auto')
            )
            self.widgets.append((min_text, max_text, last_text))
        # attach them to the display
        for min_text, max_text, last_text in self.widgets:
            display(min_text)
            display(max_text)
            display(last_text)
        
        # create a reset button
        layout = widgets.Layout(width='auto', height='40px')
        self.reset_btn = widgets.Button(
            description=f'Reset {self.name} MinMax',
            layout=layout)
        display(self.reset_btn)
        self.reset_btn.on_click(self.reset_minmax)

    def update_minmax(self):
        self.min = np.minimum(self.min, self.features.data)
        self.max = np.maximum(self.max, self.features.data)

    def update_widget(self):
        for i, (min_text, max_text, last_text) in enumerate(self.widgets):
            min_text.value = str(self.min[0][i])
            max_text.value = str(self.max[0][i])
            last_text.value = str(self.features.data[0][i])

    def update(self):
        self.update_minmax()
        self.update_widget()

    def reset_minmax(self, _ = None):
        self.min = np.zeros_like(self.features.data) * 1e6
        self.max = np.zeros_like(self.features.data) * -1e6
        self.update_widget()


class MeanPixelValue(Feature):
    """Compute the mean pixel value within a probe."""
    def __init__(self):
        super().__init__(num_features=1, name="MeanPixelValue")

    def compute(self, mat):
        return np.mean(mat)