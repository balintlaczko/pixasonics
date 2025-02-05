import numpy as np
import signalflow as sf
from .utils import samps2mix, broadcast_params, array2str
from .ui import SynthCard, find_widget_by_tag

class Theremin(sf.Patch):
    def __init__(self, frequency=440, amplitude=0.5, panning=0):
        super().__init__()
        self.input_buffers = {}
        self.params = {
            "frequency": {
                "min": 60,
                "max": 4000,
                "default": 440,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
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
                "buffer_player": None,
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
                "buffer_player": None,
                "name": "Theremin Panning",
                "param_name": "panning",
                "owner": self
            }
        }
        self.frequency, self.amplitude, self.panning = broadcast_params(frequency, amplitude, panning)
        self.num_channels = len(self.frequency) # at this point all lengths are the same
        print(f"Theremin with {self.num_channels} channels")
        self.smooth_n_samps = 24000

        self.frequency_buffer = sf.Buffer(self.num_channels, 1)
        self.amplitude_buffer = sf.Buffer(self.num_channels, 1)
        self.panning_buffer = sf.Buffer(self.num_channels, 1)
        
        self.frequency_buffer.data[:, :] = np.array(self.frequency).reshape(self.num_channels, 1)
        self.params["frequency"]["buffer"] = self.frequency_buffer
        self.params["frequency"]["default"] = self.frequency
        
        self.amplitude_buffer.data[:, :] = np.array(self.amplitude).reshape(self.num_channels, 1)
        self.params["amplitude"]["buffer"] = self.amplitude_buffer
        self.params["amplitude"]["default"] = self.amplitude

        self.panning_buffer.data[:, :] = np.array(self.panning).reshape(self.num_channels, 1)
        self.params["panning"]["buffer"] = self.panning_buffer
        self.params["panning"]["default"] = self.panning
        
        self.frequency_value = sf.BufferPlayer(self.frequency_buffer, loop=True)
        self.params["frequency"]["buffer_player"] = self.frequency_value
        self.amplitude_value = sf.BufferPlayer(self.amplitude_buffer, loop=True)
        self.params["amplitude"]["buffer_player"] = self.amplitude_value
        self.panning_value = sf.BufferPlayer(self.panning_buffer, loop=True)
        self.params["panning"]["buffer_player"] = self.panning_value
        
        freq_smooth = sf.Smooth(self.frequency_value, samps2mix(self.smooth_n_samps))
        amplitude_smooth = sf.Smooth(self.amplitude_value, samps2mix(self.smooth_n_samps))
        panning_smooth = sf.Smooth(self.panning_value, samps2mix(self.smooth_n_samps)) # still between -1 and 1
        
        sine = sf.SineOscillator(freq_smooth)
        # output = sf.StereoPanner(sine * amplitude_smooth, pan=panning_smooth)
        output = Mixer(sine * amplitude_smooth, panning_smooth * 0.5 + 0.5, out_channels=2) # pan all channels in a stereo space with the pansig scaled between 0 and 1
        
        self.set_output(output)

        self.id = str(id(self))
        self.create_ui()

    def set_input_buf(self, name, value, from_slider=False):
        # print(f"Setting {name} to {value}, from_slider={from_slider}")
        self.params[name]["buffer"].data[:, :] = value
        if not from_slider:
            slider = find_widget_by_tag(self.ui, name)
            # TODO: avoid double setting here (when slider.value changes it will also call set_input_buf)
            slider.value = value if self.num_channels == 1 else array2str(value)

    def reset_to_default(self):
        for param in self.params:
            self.set_input_buf(param, np.array(self.params[param]["default"]).reshape(self.num_channels, 1), from_slider=False)

    def __getitem__(self, key):
        return self.params[key]
    
    def create_ui(self):
        self._ui = SynthCard(
            name="Theremin",
            id=self.id,
            params=self.params,
            num_channels=self.num_channels
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


class Mixer(sf.Patch):
    def __init__(self, input_sig, pan_sig, out_channels=2):
        super().__init__()
        assert input_sig.num_output_channels == pan_sig.num_output_channels
        n = input_sig.num_output_channels
        panner = [sf.ChannelPanner(out_channels, input_sig[i] / n, pan_sig[i]) for i in range(n)]
        _sum = sf.Sum(panner)
        self.set_output(_sum)


class UpMixer(sf.Patch):
    def __init__(self, input_sig, out_channels=5):
        super().__init__()
        n = input_sig.num_output_channels # e.g. 2
        output_x = np.linspace(0, n-1, out_channels) # e.g. [0, 0.25, 0.5, 0.75, 1]
        output_y = output_x * (out_channels - 1) # e.g. [0, 1, 2, 3, 4]
        upmixed_list = [sf.WetDry(input_sig[int(output_i)], input_sig[int(output_i) + 1], float(output_i - int(output_i))) for output_i in output_x[:-1]]
        upmixed_list.append(input_sig[n-1])
        expanded_list = [sf.ChannelPanner(out_channels, upmixed_list[i], float(output_y[i])) for i in range(out_channels)]
        _out = sf.Sum(expanded_list)
        self.set_output(_out)