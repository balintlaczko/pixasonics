import numpy as np
import signalflow as sf
from .ui import FeatureCard, find_widget_by_tag


class Feature():
    """Feature base class"""
    def __init__(self, num_features=1, name="Feature"):
        self.num_features = num_features
        self.name = name
        self.features = sf.Buffer(1, num_features)
        self.min = np.ones_like(self.features.data) * 1e6
        self.max = np.ones_like(self.features.data) * -1e6
        self.id = str(id(self))
        self.create_ui()

    def __call__(self, mat):
        self.features.data[:, :] = self.compute(mat)
        self.update()
    
    def compute(self, mat):
        return NotImplemented

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
    
    def array2str(self, arr, decimals=3):
        """String from an array, where elements are rounded to decimals, and the square brackets are removed."""
        return str(np.round(arr, decimals)).replace('[', '').replace(']', '')

    def update_minmax(self):
        self.min = np.minimum(self.min, self.features.data)
        self.max = np.maximum(self.max, self.features.data)

    def update_ui(self):
        self._ui_min.value = self.array2str(self.min)
        self._ui_max.value = self.array2str(self.max)
        self._ui_value.value = self.array2str(self.features.data) 

    def update(self):
        self.update_minmax()
        self.update_ui()

    def reset_minmax(self, _ = None):
        self.min = np.ones_like(self.features.data) * 1e6
        self.max = np.ones_like(self.features.data) * -1e6
        self.update_ui()


class MeanPixelValue(Feature):
    """Compute the mean pixel value within a probe."""
    def __init__(self):
        super().__init__(num_features=1, name="Mean Pixel Value")

    def compute(self, mat):
        return np.mean(mat)