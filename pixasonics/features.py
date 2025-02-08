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


# Channel-based abstractions

class MeanChannelValue(Feature):
    """Compute the mean channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MeanChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="mean",
            name=name
        )

class MedianChannelValue(Feature):
    """Compute the median channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MedianChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="median",
            name=name
        )

class MaxChannelValue(Feature):
    """Compute the max channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MaxChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="max",
            name=name
        )

class MinChannelValue(Feature):
    """Compute the min channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MinChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="min",
            name=name
        )

class SumChannelValue(Feature):
    """Compute the sum channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="SumChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="sum",
            name=name
        )

class StdChannelValue(Feature):
    """Compute the std channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="StdChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="std",
            name=name
        )

class VarChannelValue(Feature):
    """Compute the var channel value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="VarChannelValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=2, # channel dim
            reduce_method="var",
            name=name
        )

# Layer-based abstractions

class MeanLayerValue(Feature):
    """Compute the mean layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MeanLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="mean",
            name=name
        )

class MedianLayerValue(Feature):
    """Compute the median layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MedianLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="median",
            name=name
        )

class MaxLayerValue(Feature):
    """Compute the max layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MaxLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="max",
            name=name
        )

class MinLayerValue(Feature):
    """Compute the min layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MinLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="min",
            name=name
        )

class SumLayerValue(Feature):
    """Compute the sum layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="SumLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="sum",
            name=name
        )

class StdLayerValue(Feature):
    """Compute the std layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="StdLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="std",
            name=name
        )

class VarLayerValue(Feature):
    """Compute the var layer value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="VarLayerValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=3, # layer dim
            reduce_method="var",
            name=name
        )

# Row-based abstractions

class MeanRowValue(Feature):
    """Compute the mean row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MeanRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="mean",
            name=name
        )

class MedianRowValue(Feature):
    """Compute the median row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MedianRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="median",
            name=name
        )

class MaxRowValue(Feature):
    """Compute the max row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MaxRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="max",
            name=name
        )

class MinRowValue(Feature):
    """Compute the min row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MinRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="min",
            name=name
        )

class SumRowValue(Feature):
    """Compute the sum row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="SumRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="sum",
            name=name
        )

class StdRowValue(Feature):
    """Compute the std row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="StdRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="std",
            name=name
        )

class VarRowValue(Feature):
    """Compute the var row value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="VarRowValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=0, # row dim
            reduce_method="var",
            name=name
        )

# Column-based abstractions

class MeanColumnValue(Feature):
    """Compute the mean column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MeanColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="mean",
            name=name
        )

class MedianColumnValue(Feature):
    """Compute the median column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MedianColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="median",
            name=name
        )

class MaxColumnValue(Feature):
    """Compute the max column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MaxColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="max",
            name=name
        )

class MinColumnValue(Feature):
    """Compute the min column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="MinColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="min",
            name=name
        )

class SumColumnValue(Feature):
    """Compute the sum column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="SumColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="sum",
            name=name
        )

class StdColumnValue(Feature):
    """Compute the std column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="StdColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="std",
            name=name
        )

class VarColumnValue(Feature):
    """Compute the var column value within a probe."""
    def __init__(
            self, 
            filter_rows=None,
            filter_columns=None,
            filter_channels=None,
            filter_layers=None,
            name="VarColumnValue"):
        super().__init__(
            filter_rows=filter_rows,
            filter_columns=filter_columns,
            filter_channels=filter_channels,
            filter_layers=filter_layers,
            target_dim=1, # column dim
            reduce_method="var",
            name=name
        )
