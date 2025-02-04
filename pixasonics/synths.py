import signalflow as sf
from .utils import samps2mix
from .ui import SynthCard, find_widget_by_tag

class Theremin(sf.Patch):
    def __init__(self, frequency=440, amplitude=0.5, smooth_n_samps=24000):
        super().__init__()
        self.input_buffers = {}
        self.params = {
            "frequency": {
                "min": 40,
                "max": 8000,
                "default": 440,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "name": "Theremin Frequency",
                "param_name": "frequency",
                "owner": self
            },
            "amplitude": {
                "min": 0,
                "max": 1,
                "default": 0.5,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "name": "Theremin Amplitude",
                "param_name": "amplitude",
                "owner": self
            },
            "panning": {
                "min": -1,
                "max": 1,
                "default": 0,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "name": "Theremin Panning",
                "param_name": "panning",
                "owner": self
            }
        }

        buf_frequency = sf.Buffer(1, 1)
        buf_amplitude = sf.Buffer(1, 1)
        buf_panning = sf.Buffer(1, 1)
        
        buf_frequency.data[0][0] = frequency
        self.params["frequency"]["buffer"] = buf_frequency
        
        buf_amplitude.data[0][0] = amplitude
        self.params["amplitude"]["buffer"] = buf_amplitude

        buf_panning.data[0][0] = 0
        self.params["panning"]["buffer"] = buf_panning
        
        frequency_value = sf.BufferPlayer(buf_frequency, loop=True)
        amplitude_value = sf.BufferPlayer(buf_amplitude, loop=True)
        panning_value = sf.BufferPlayer(buf_panning, loop=True)
        
        freq_smooth = sf.Smooth(frequency_value, samps2mix(smooth_n_samps))
        amplitude_smooth = sf.Smooth(amplitude_value, samps2mix(smooth_n_samps))
        panning_smooth = sf.Smooth(panning_value, samps2mix(smooth_n_samps))
        
        sine = sf.SineOscillator(freq_smooth)
        output = sf.StereoPanner(sine * amplitude_smooth, pan=panning_smooth)
        
        self.set_output(output)

        self.id = str(id(self))
        self.create_ui()

    def set_input_buf(self, name, value, from_slider=False):
        # print(f"Setting {name} to {value}, from_slider={from_slider}")
        self.params[name]["buffer"].data[0][0] = value
        if not from_slider:
            slider = find_widget_by_tag(self.ui, name)
            slider.value = value

    def reset_to_default(self):
        for param in self.params:
            self.set_input_buf(param, self.params[param]["default"], from_slider=False)

    def __getitem__(self, key):
        return self.params[key]
    
    def create_ui(self):
        self._ui = SynthCard(
            name="Theremin",
            id=self.id,
            params=self.params
        )
        self._ui.synth = self
        # # set up slider callbacks
        # for param in self.params:
        #     slider = find_widget_by_tag(self.ui, param)
        #     slider.observe(
        #         lambda change: self.set_input_buf(
        #             change["owner"].tag, 
        #             change["new"],
        #             from_slider=True
        #         ), 
        #         names="value")

    @property
    def ui(self):
        return self._ui()
    
    def __repr__(self):
        return f"Theremin {self.id}"


class InputTestParent(sf.Patch):
    def __init__(self, hello=0):
        super().__init__()
        hello = self.add_input("hello", hello)
        input_test = InputTest()
        input_test.play()
        input_test.set_input("hello", hello)
        self.set_output(input_test)

class InputTest(sf.Patch):
    def __init__(self, hello=0):
        super().__init__()
        hello = self.add_input("hello", hello)
        self.set_output(hello)


class LinearSmooth(sf.Patch):
    def __init__(self, input_signal, smooth_n_samps=480, sr=48000):
        super().__init__()
        self.history = sf.Constant(0)
        self.diff_signal = input_signal - self.history
        self.smooth_coeff = 1 / smooth_n_samps
        # self.history = self.history + (diff_signal * smooth_coeff)
        interp = self.history + (self.diff_signal * self.smooth_coeff)
        self.history.set_value(interp.output_buffer[0][-1])
        self.set_output(self.history)
