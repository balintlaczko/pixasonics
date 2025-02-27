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
            reduce_method="mean", # can be "mean", "max", "min", "sum", "std", "var", "median"
            name="Feature"):
        # Init private attrs
        self._filter_rows = None
        self._filter_columns = None
        self._filter_channels = None
        self._filter_layers = None
        self._target_dim = None
        self._reduce_method = None
        self._name = None
        self._app = None
        self._id = None
        # Call setters
        self.filter_rows = filter_rows
        self.filter_columns = filter_columns
        self.filter_channels = filter_channels
        self.filter_layers = filter_layers
        self.target_dim = target_dim
        self.reduce_method = reduce_method
        self.name = name
        self.id = str(id(self))
        # Init public attrs
        self.features = sf.Buffer(1, 1) # default to 1 feature/channel
        self.min = np.ones_like(self.features.data) * 1e6
        self.max = np.ones_like(self.features.data) * -1e6
        # Create the UI card
        self.create_ui()

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id):
        assert isinstance(id, str), "id must be a string"
        self._id = id
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        assert isinstance(name, str), "name must be a string"
        self._name = name
    
    @property
    def filter_rows(self):
        return self._filter_rows
    
    @filter_rows.setter
    def filter_rows(self, filter_rows):
        if filter_rows is not None:
            assert isinstance(filter_rows, (int, slice, list, str)), "filter_rows must be int, slice, list, str or None"
        self._filter_rows = filter_rows

    @property
    def filter_columns(self):
        return self._filter_columns
    
    @filter_columns.setter
    def filter_columns(self, filter_columns):
        if filter_columns is not None:
            assert isinstance(filter_columns, (int, slice, list, str)), "filter_columns must be int, slice, list, str or None"
        self._filter_columns = filter_columns

    @property
    def filter_channels(self):
        return self._filter_channels
    
    @filter_channels.setter
    def filter_channels(self, filter_channels):
        if filter_channels is not None:
            assert isinstance(filter_channels, (int, slice, list, str)), "filter_channels must be int, slice, list, str or None"
        self._filter_channels = filter_channels

    @property
    def filter_layers(self):
        return self._filter_layers
    
    @filter_layers.setter
    def filter_layers(self, filter_layers):
        if filter_layers is not None:
            assert isinstance(filter_layers, (int, slice, list, str)), "filter_layers must be int, slice, list, str or None"
        self._filter_layers = filter_layers

    @property
    def target_dim(self):
        return self._target_dim
    
    @target_dim.setter
    def target_dim(self, target_dim):
        assert target_dim in [0, 1, 2, 3], "target_dim must be 0, 1, 2, or 3"
        self._target_dim = target_dim

    @property
    def app(self):
        return self._app
    
    @app.setter
    def app(self, app):
        self._app = app
        self._process_image(app.bg_hires)

    @property
    def reduce_method(self):
        return self._reduce_method
    
    @reduce_method.setter
    def reduce_method(self, reduce_method):
        assert reduce_method in ["mean", "max", "min", "sum", "std", "var", "median"], \
            "Unknown reduce method string. Must be one of: mean, max, min, sum, std, var, median"
        self._reduce_method = reduce_method

    @property 
    def reduce(self):
        """Get the reduction function based on reduce_method string"""
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
        
    @property
    def reduce_axis(self):
        return tuple(i for i in range(4) if i != self.target_dim)

    def __call__(self, mat):
        mat_filtered = filter_matrix(
            mat,
            self.filter_rows,
            self.filter_columns,
            self.filter_channels,
            self.filter_layers
        )
        computed = self.compute(mat_filtered)
        # Here we need to assert that the computed shape is 1D (num_features,)
        assert len(computed.shape) == 1, f"Computed shape is not 1D: {computed.shape}"
        if computed.shape[0] != self.num_features:
            # self.initialize(mat_filtered)
            self.initialize(computed) # TODO: this means it will always take the local minmax
        self.features.data[:, :] = computed[..., None] # add the sample dimension, so it is (num_features, 1)

        self.update_minmax() # then we have to keep a running minmax
        self.update_ui()
    
    def compute(self, mat):
        """Compute the feature from the matrix, override this method for custom computation.
        The custom computation should return a 1D array of shape (num_features,)
        """
        return self.reduce(mat, axis=self.reduce_axis)
    
    
    def process_image(self, mat):
        """Override this method for custom processing of the App's whole image, called upon attachment to the App.
        The custom processing should return a matrix of the same shape as the input.
        This will be used to calculate the number of features, and the min and max values along the axis defined by self.target_dim.
        """
        return mat


    def _process_image(self, mat):
        """This is called when the Feature is attached to an App, where the App's full image is passed in.
        Override the process_image method for custom processing.
        Here we filter the matrix based on all self.filter_* attributes, 
        process the image with self.process_image, then compute min and max, initialize the feature Buffer,
        and update the ui card.
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
        mat_processed = self.process_image(mat_filtered)
        
        self.initialize(mat_processed)
        self.update_ui()


    def initialize(self, mat):
        if len(mat.shape) == 4:
            self.min = np.min(mat, axis=self.reduce_axis)[..., None]
            self.max = np.max(mat, axis=self.reduce_axis)[..., None]
            self.num_features = self.min.shape[0]
        else:
            self.min = np.min(mat)[..., None]
            self.max = np.max(mat)[..., None]
            self.num_features = mat.shape[0]
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
