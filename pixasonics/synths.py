import signalflow as sf
from .utils import samps2mix

class Theremin(sf.Patch):
    def __init__(self, frequency=440, amplitude=0.5, smooth_n_samps=24000):
        super().__init__()
        self.input_buffers = {}
        self.params = {
            "frequency": {
                "min": 20,
                "max": 20000,
                "default": 440,
                "unit": "Hz",
                "buffer": None
            },
            "amplitude": {
                "min": 0,
                "max": 1,
                "default": 0.5,
                "unit": "",
                "buffer": None
            }
        }

        buf_frequency = sf.Buffer(1, 1)
        buf_amplitude = sf.Buffer(1, 1)
        
        buf_frequency.data[0][0] = frequency
        self.params["frequency"]["buffer"] = buf_frequency
        
        buf_amplitude.data[0][0] = amplitude
        self.params["amplitude"]["buffer"] = buf_amplitude
        
        frequency_value = sf.BufferPlayer(buf_frequency, loop=True)
        amplitude_value = sf.BufferPlayer(buf_amplitude, loop=True)
        
        freq_smooth = sf.Smooth(frequency_value, samps2mix(smooth_n_samps))
        amplitude_smooth = sf.Smooth(amplitude_value, samps2mix(smooth_n_samps))
        
        sine = sf.SineOscillator(freq_smooth)
        output = sf.StereoPanner(sine * amplitude_smooth, pan=0)
        
        self.set_output(output)

    def set_input_buf(self, name, value):
        self.params[name]["buffer"].data[0][0] = value

    def __getitem__(self, key):
        return self.params[key]