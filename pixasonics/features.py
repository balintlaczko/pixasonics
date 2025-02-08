import numpy as np
import signalflow as sf
from .ui import FeatureCard, find_widget_by_tag
from .utils import array2str, filter_matrix


class Feature():
    """Feature base class"""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            target_dim=2, # channel dim by default
            reduce_method="mean", # can be "mean", "max", "min", "sum", "std", "var", "median", 
            name="Value"):
        self.filter_rows = filter_rows
        self.filter_columns = filter_columns
        self.filter_channels = filter_channels
        self.filter_layers = filter_layers

        self.target_dim = target_dim
        assert self.target_dim in [0, 1, 2, 3], "target_dim must be 0, 1, 2, or 3"
        
        self.reduce_method = reduce_method
        assert self.reduce_method in ["mean", "max", "min", "sum", "std", "var", "median"], "Unknown reduce method"
        
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

    @property
    def reduce(self):
        if self.reduce_method == "mean":
            return np.mean
        elif self.reduce_method == "max":
            return np.max
        elif self.reduce_method == "min":
            return np.min
        elif self.reduce_method == "sum":
            return np.sum
        elif self.reduce_method == "std":
            return np.std
        elif self.reduce_method == "var":
            return np.var
        elif self.reduce_method == "median":
            return np.median
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")
        
    @property
    def axis(self):
        return tuple(i for i in range(4) if i != self.target_dim)

    def __call__(self, mat):
        # print("In Feature:", self.features.data.shape)
        mat_filtered = filter_matrix(
            mat,
            self.filter_rows,
            self.filter_columns,
            self.filter_channels,
            self.filter_layers
        )
        computed = self.compute(mat_filtered)
        if computed.shape[0] != self.num_features:
            self.initialize(mat_filtered)
        self.features.data[:, :] = self.compute(mat_filtered)

        # if self.target_dim < 2: # when the target dim is either width or height
        self.update_minmax() # then we have to keep a running minmax
        self.update_ui()
    
    def compute(self, mat):
        return self.reduce(mat, axis=self.axis)[..., None]
    
    
    def process_image(self, mat):
        """By default, this will compute global min and max. Override this method for custom processing.
        Image arrays are assumed to have a shape of (H, W, C, L) or (H, W, C) where H is height, 
        W is width, C is channels, and L is layer.
        """
        mat_filtered = filter_matrix(
            mat,
            self.filter_rows,
            self.filter_columns,
            self.filter_channels,
            self.filter_layers
        )
        self.initialize(mat_filtered)
        self.update_ui()


    def initialize(self, mat):
        self.min = np.min(mat, axis=self.axis)[..., None]
        self.max = np.max(mat, axis=self.axis)[..., None]
        # set number of features to the number of channels
        self.num_features = self.min.shape[0]
        self.features = sf.Buffer(self.num_features, 1)

        
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
        self._ui_num_features = find_widget_by_tag(self._ui(), "num_features")

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
        self._ui_num_features.value = str(self.num_features)

    def update(self):
        self.update_minmax()
        self.update_ui()

    def reset_minmax(self, _ = None):
        self.min = np.ones_like(self.features.data) * 1e6
        self.max = np.ones_like(self.features.data) * -1e6
        self.update_minmax()
        self.update_ui()


# class MeanPixelValue(Feature):
#     """Compute the mean pixel value within a probe."""
#     def __init__(self, selected_channels=None):
#         super().__init__(selected_channels=selected_channels, name="Mean Pixel Value")

#     def compute(self, mat):
#         # if 3D
#         if len(mat.shape) == 3:
#             return np.mean(mat, axis=(0, 1))[..., None]
#         elif len(mat.shape) == 4:
#             return np.mean(mat, axis=(0, 1, 3))[..., None]
#         else:
#             raise ValueError("Input must be 3D or 4D")