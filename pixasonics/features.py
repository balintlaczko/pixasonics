import numpy as np
import signalflow as sf
from .ui import FeatureCard, find_widget_by_tag
from .utils import array2str


class Feature():
    """Feature base class"""
    def __init__(self, selected_channels=None, name="Feature"):
        self.selected_channels = selected_channels
        self.name = name
        self.features = sf.Buffer(1, 1) # default to 1 feature/channel
        self.min = np.ones_like(self.features.data) * 1e6
        self.max = np.ones_like(self.features.data) * -1e6

        self._app = None

        self.id = str(id(self))
        self.create_ui()

    @property
    def app(self):
        return self._app
    
    @app.setter
    def app(self, app):
        self._app = app
        self.process_image(app.bg_hires)

    def __call__(self, mat):
        # print("In Feature:", self.features.data.shape)
        mat_filtered = self.filter_selected_channels(mat, self.selected_channels)
        self.features.data[:, :] = self.compute(mat_filtered)
        self.update_ui()
    
    def compute(self, mat):
        return NotImplemented
    
    def filter_selected_channels(self, mat, channels=None):
        if channels is not None:
            # if 3D
            if len(mat.shape) == 3:
                return mat[:, :, channels]
            # if 4D
            elif len(mat.shape) == 4:
                return mat[:, :, channels, :]
        else:
            return mat
    
    def process_image(self, mat):
        """By default, this will compute global min and max. Override this method for custom processing.
        Image arrays are assumed to have a shape of (H, W, C, L) or (H, W, C) where H is height, 
        W is width, C is channels, and L is layer.
        """
        mat_filtered = self.filter_selected_channels(mat, self.selected_channels)
        # if 3d
        if len(mat_filtered.shape) == 3:
            self.min = np.min(mat_filtered, axis=(0, 1)) # reduce H and W
            self.max = np.max(mat_filtered, axis=(0, 1))
        # if 4d
        elif len(mat_filtered.shape) == 4:
            self.min = np.min(mat_filtered, axis=(0, 1, 3)) # reduce H, W, L
            self.max = np.max(mat_filtered, axis=(0, 1, 3))
        else:
            raise ValueError("Input must be 3D or 4D")
        # add a new axis to the min and max arrays
        self.min = self.min[..., None]
        self.max = self.max[..., None]
        
        # set number of features to the number of channels
        self.num_features = self.min.shape[0]
        self.features = sf.Buffer(self.num_features, 1)

        self.update_ui()

    def create_ui(self):
        self._ui = FeatureCard(
            name=self.name,
            id=self.id,
            min=str(self.min),
            max=str(self.max),
            value=str(self.features.data),
        )
        self._ui.feature = self
        self._ui_min = find_widget_by_tag(self._ui(), "min")
        self._ui_max = find_widget_by_tag(self._ui(), "max")
        self._ui_value = find_widget_by_tag(self._ui(), "value")

    @property
    def ui(self):
        return self._ui()
    
    def __repr__(self):
        return f"Feature {self.id}: {self.name}"

    def update_minmax(self):
        self.min = np.minimum(self.min, self.features.data)
        self.max = np.maximum(self.max, self.features.data)

    def update_ui(self):
        self._ui_min.value = array2str(self.min)
        self._ui_max.value = array2str(self.max)
        self._ui_value.value = array2str(self.features.data) 

    def update(self):
        self.update_minmax()
        self.update_ui()

    # def reset_minmax(self, _ = None):
    #     self.min = np.ones_like(self.features.data) * 1e6
    #     self.max = np.ones_like(self.features.data) * -1e6
    #     self.update_ui()


class MeanPixelValue(Feature):
    """Compute the mean pixel value within a probe."""
    def __init__(self, selected_channels=None):
        super().__init__(selected_channels=selected_channels, name="Mean Pixel Value")

    def compute(self, mat):
        # if 3D
        if len(mat.shape) == 3:
            return np.mean(mat, axis=(0, 1))[..., None]
        elif len(mat.shape) == 4:
            return np.mean(mat, axis=(0, 1, 3))[..., None]
        else:
            raise ValueError("Input must be 3D or 4D")