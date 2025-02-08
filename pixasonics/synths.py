import numpy as np
import signalflow as sf
from .utils import samps2mix, broadcast_params, array2str
from .ui import SynthCard, EnvelopeCard, find_widget_by_tag

class Theremin(sf.Patch):
    def __init__(self, frequency=440, amplitude=0.5, panning=0, name="Theremin"):
        super().__init__()
        self.name = name
        self.params = {
            "frequency": {
                "min": 60,
                "max": 4000,
                "default": 440,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Frequency",
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
                "name": f"{self.name} Amplitude",
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
                "name": f"{self.name} Panning",
                "param_name": "panning",
                "owner": self
            }
        }
        self.frequency, self.amplitude, self.panning = broadcast_params(frequency, amplitude, panning)
        self.num_channels = len(self.frequency) # at this point all lengths are the same
        print(f"Theremin {self.name} with {self.num_channels} channels")

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
        
        # smooth_time = 0.05
        self.smooth_n_samps = 24000
        mix_val = samps2mix(self.smooth_n_samps)
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        freq_smooth = sf.Smooth(self.frequency_value, mix_val)
        # freq_smooth = LinearSmooth(self.frequency_value, smooth_time=smooth_time)
        amplitude_smooth = sf.Smooth(self.amplitude_value, mix_val)
        # amplitude_smooth = LinearSmooth(self.amplitude_value, smooth_time=smooth_time)
        panning_smooth = sf.Smooth(self.panning_value, mix_val) # still between -1 and 1
        # panning_smooth = LinearSmooth(self.panning_value, smooth_time=smooth_time)
        
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
            name=self.name,
            id=self.id,
            params=self.params,
            num_channels=self.num_channels
        )
        self._ui.synth = self

    @property
    def ui(self):
        return self._ui()
    
    def __repr__(self):
        return f"Theremin {self.id}: {self.name}"
    

class Envelope(sf.Patch):
    def __init__(self, attack=0.01, decay=0.01, sustain=0.5, release=0.1):
        super().__init__()
        self.params = {
            "attack": {
                "min": 0.001,
                "max": 3600,
                "default": 0.01,
                "step" : 0.01,
                "param_name": "attack"
            },
            "decay": {
                "min": 0.001,
                "max": 3600,
                "default": 0.01,
                "step" : 0.01,
                "param_name": "decay"
            },
            "sustain": {
                "min": 0,
                "max": 1,
                "default": 0.5,
                "step": 0.1,
                "param_name": "sustain"
            },
            "release": {
                "min": 0.001,
                "max": 3600,
                "default": 0.1,
                "step" : 0.1,
                "param_name": "release"
            }
        }
        self.params["attack"]["default"] = attack
        self.params["decay"]["default"] = decay
        self.params["sustain"]["default"] = sustain
        self.params["release"]["default"] = release

        for param in self.params.keys():
            self.params[param]["value"] = self.params[param]["default"]

        gate = self.add_input("gate", 0)
        attack = self.add_input("attack", self.params["attack"]["default"])
        decay = self.add_input("decay", self.params["decay"]["default"])
        sustain = self.add_input("sustain", self.params["sustain"]["default"])
        release = self.add_input("release", self.params["release"]["default"])

        adsr = sf.ADSREnvelope(
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            gate=gate
        )

        asr = sf.ASREnvelope(
            attack=attack,
            sustain=sustain,
            release=release,
            clock=0
        )

        self.set_trigger_node(asr)
        self.set_output(adsr + asr)

        self.id = str(id(self))
        self.create_ui()

    def on(self):
        self.set_input("gate", 1)

    def off(self):
        self.set_input("gate", 0)

    def __getitem__(self, key):
        return self.params[key]
    
    def create_ui(self):
        self._ui = EnvelopeCard("Envelope", self.id, self.params)
        self._ui.envelope = self

    @property
    def ui(self):
        return self._ui()
    
    def set_param_from_ui(self, param_name, value):
        self.params[param_name]["value"] = value
        self.set_input(param_name, value)
    
    @property
    def attack(self):
        return self.params["attack"]["value"]
    
    @attack.setter
    def attack(self, value):
        self.params["attack"]["value"] = value
        self.set_input("attack", value)
        self._ui.attack = value

    @property
    def decay(self):
        return self.params["decay"]["value"]
    
    @decay.setter
    def decay(self, value):
        self.params["decay"]["value"] = value
        self.set_input("decay", value)
        self._ui.decay = value

    @property
    def sustain(self):
        return self.params["sustain"]["value"]
    
    @sustain.setter
    def sustain(self, value):
        self.params["sustain"]["value"] = value
        self.set_input("sustain", value)
        self._ui.sustain = value

    @property
    def release(self):
        return self.params["release"]["value"]
    
    @release.setter
    def release(self, value):
        self.params["release"]["value"] = value
        self.set_input("release", value)
        self._ui.release = value


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


class LinearSmooth(sf.Patch):
    def __init__(self, input_sig, smooth_time=0.1):
        super().__init__()
        graph = sf.AudioGraph.get_shared_graph()
        samps = graph.sample_rate * smooth_time
        steps = samps / graph.output_buffer_size
        steps = sf.If(steps < 1, 1, steps)

        current_value_buf = sf.Buffer(1, graph.output_buffer_size)
        current_value = sf.FeedbackBufferReader(current_value_buf)

        history_buf = sf.Buffer(1, graph.output_buffer_size)
        history = sf.FeedbackBufferReader(history_buf)

        change = input_sig != history
        target = sf.SampleAndHold(input_sig, change)
        diff = sf.SampleAndHold(target - current_value, change)

        increment = diff / steps

        out = sf.If(sf.Abs(target - current_value) < sf.Abs(increment), target, current_value + increment)
        graph.add_node(sf.HistoryBufferWriter(current_value_buf, out))
        graph.add_node(sf.HistoryBufferWriter(history_buf, input_sig))
        self.set_output(out)