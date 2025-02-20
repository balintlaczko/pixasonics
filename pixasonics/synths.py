import numpy as np
import signalflow as sf
from .utils import samps2mix, broadcast_params, array2str, ParamSliderDebouncer
from .ui import SynthCard, EnvelopeCard, find_widget_by_tag

PARAM_SLIDER_DEBOUNCE_TIME = 0.05

class Synth(sf.Patch):
    def __init__(self):
        super().__init__()


class Theremin(Synth):
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
        amplitude_smooth = sf.Smooth(self.amplitude_value, mix_val)
        panning_smooth = sf.Smooth(self.panning_value, mix_val) # still between -1 and 1
        
        sine = sf.SineOscillator(freq_smooth)
        output = Mixer(sine * amplitude_smooth, panning_smooth * 0.5 + 0.5, out_channels=2) # pan all channels in a stereo space with the pansig scaled between 0 and 1
        
        self.set_output(output)

        self.id = str(id(self))
        self.create_ui()

        self.debouncer = ParamSliderDebouncer(PARAM_SLIDER_DEBOUNCE_TIME) if self.num_channels == 1 else None

    def set_input_buf(self, name, value, from_slider=False):
        self.params[name]["buffer"].data[:, :] = value
        if not from_slider and self.num_channels == 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.unobserve_all()
            slider_value = value if self.num_channels == 1 else array2str(value)
            self.debouncer.submit(name, lambda: self.update_slider(slider, slider_value))
        elif not from_slider and self.num_channels > 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.value = array2str(value)

    def update_slider(self, slider, value):
        slider.unobserve_all()
        slider.value = value
        slider.observe(
            lambda change: self.set_input_buf(
                    change["owner"].tag, 
                    change["new"],
                    from_slider=True
                ), 
                names="value")
        

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
    

class Oscillator(Synth):
    def __init__(
            self, 
            frequency=440, 
            amplitude=0.5, 
            panning=0, 
            lp_cutoff=20000,
            lp_resonance=0.5,
            hp_cutoff=20,
            hp_resonance=0.5,
            waveform="sine",
            name="Oscillator"):
        super().__init__()

        wf_types = ["sine", "square", "saw", "triangle"]
        assert waveform in wf_types, f"Waveform must be one of {wf_types}"
        self.waveform = waveform

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
            },
            "lp_cutoff": {
                "min": 20,
                "max": 20000,
                "default": 20000,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} LP Cutoff",
                "param_name": "lp_cutoff",
                "owner": self
            },
            "lp_resonance": {
                "min": 0,
                "max": 0.999,
                "default": 0,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} LP Resonance",
                "param_name": "lp_resonance",
                "owner": self
            },
            "hp_cutoff": {
                "min": 20,
                "max": 20000,
                "default": 20,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} HP Cutoff",
                "param_name": "hp_cutoff",
                "owner": self
            },
            "hp_resonance": {
                "min": 0,
                "max": 0.999,
                "default": 0,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} HP Resonance",
                "param_name": "hp_resonance",
                "owner": self
            },
        }
        self.frequency, self.amplitude, self.panning, self.lp_cutoff, self.lp_resonance, self.hp_cutoff, self.hp_resonance = broadcast_params(frequency, amplitude, panning, lp_cutoff, lp_resonance, hp_cutoff, hp_resonance)
        self.num_channels = len(self.frequency) # at this point all lengths are the same

        self.frequency_buffer = sf.Buffer(self.num_channels, 1)
        self.amplitude_buffer = sf.Buffer(self.num_channels, 1)
        self.panning_buffer = sf.Buffer(self.num_channels, 1)
        self.lp_cutoff_buffer = sf.Buffer(self.num_channels, 1)
        self.lp_resonance_buffer = sf.Buffer(self.num_channels, 1)
        self.hp_cutoff_buffer = sf.Buffer(self.num_channels, 1)
        self.hp_resonance_buffer = sf.Buffer(self.num_channels, 1)
        
        self.frequency_buffer.data[:, :] = np.array(self.frequency).reshape(self.num_channels, 1)
        self.params["frequency"]["buffer"] = self.frequency_buffer
        self.params["frequency"]["default"] = self.frequency
        
        self.amplitude_buffer.data[:, :] = np.array(self.amplitude).reshape(self.num_channels, 1)
        self.params["amplitude"]["buffer"] = self.amplitude_buffer
        self.params["amplitude"]["default"] = self.amplitude

        self.panning_buffer.data[:, :] = np.array(self.panning).reshape(self.num_channels, 1)
        self.params["panning"]["buffer"] = self.panning_buffer
        self.params["panning"]["default"] = self.panning

        self.lp_cutoff_buffer.data[:, :] = np.array(self.lp_cutoff).reshape(self.num_channels, 1)
        self.params["lp_cutoff"]["buffer"] = self.lp_cutoff_buffer
        self.params["lp_cutoff"]["default"] = self.lp_cutoff

        self.lp_resonance_buffer.data[:, :] = np.array(self.lp_resonance).reshape(self.num_channels, 1)
        self.params["lp_resonance"]["buffer"] = self.lp_resonance_buffer
        self.params["lp_resonance"]["default"] = self.lp_resonance

        self.hp_cutoff_buffer.data[:, :] = np.array(self.hp_cutoff).reshape(self.num_channels, 1)
        self.params["hp_cutoff"]["buffer"] = self.hp_cutoff_buffer
        self.params["hp_cutoff"]["default"] = self.hp_cutoff

        self.hp_resonance_buffer.data[:, :] = np.array(self.hp_resonance).reshape(self.num_channels, 1)
        self.params["hp_resonance"]["buffer"] = self.hp_resonance_buffer
        self.params["hp_resonance"]["default"] = self.hp_resonance
        
        self.frequency_value = sf.BufferPlayer(self.frequency_buffer, loop=True)
        self.params["frequency"]["buffer_player"] = self.frequency_value
        self.amplitude_value = sf.BufferPlayer(self.amplitude_buffer, loop=True)
        self.params["amplitude"]["buffer_player"] = self.amplitude_value
        self.panning_value = sf.BufferPlayer(self.panning_buffer, loop=True)
        self.params["panning"]["buffer_player"] = self.panning_value
        self.lp_cutoff_value = sf.BufferPlayer(self.lp_cutoff_buffer, loop=True)
        self.params["lp_cutoff"]["buffer_player"] = self.lp_cutoff_value
        self.lp_resonance_value = sf.BufferPlayer(self.lp_resonance_buffer, loop=True)
        self.params["lp_resonance"]["buffer_player"] = self.lp_resonance_value
        self.hp_cutoff_value = sf.BufferPlayer(self.hp_cutoff_buffer, loop=True)
        self.params["hp_cutoff"]["buffer_player"] = self.hp_cutoff_value
        self.hp_resonance_value = sf.BufferPlayer(self.hp_resonance_buffer, loop=True)
        self.params["hp_resonance"]["buffer_player"] = self.hp_resonance_value

        # Clip the resonance values to avoid filter instability
        self.lp_resonance_value_clip = sf.Clip(self.lp_resonance_value, 0, 0.999)
        self.hp_resonance_value_clip = sf.Clip(self.hp_resonance_value, 0, 0.999)
        
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        freq_smooth = sf.Smooth(self.frequency_value, mix_val)
        amplitude_smooth = sf.Smooth(self.amplitude_value, mix_val)
        panning_smooth = sf.Smooth(self.panning_value, mix_val) # still between -1 and 1
        lp_cutoff_smooth = sf.Smooth(self.lp_cutoff_value, mix_val)
        lp_resonance_smooth = sf.Smooth(self.lp_resonance_value_clip, mix_val)
        hp_cutoff_smooth = sf.Smooth(self.hp_cutoff_value, mix_val)
        hp_resonance_smooth = sf.Smooth(self.hp_resonance_value_clip, mix_val)
        
        osc_templates = [sf.SineOscillator, sf.SquareOscillator, sf.SawOscillator, sf.TriangleOscillator]
        osc = osc_templates[wf_types.index(self.waveform)](freq_smooth)
        lp = sf.SVFilter(osc, filter_type="low_pass", cutoff=lp_cutoff_smooth, resonance=lp_resonance_smooth)
        hp = sf.SVFilter(lp, filter_type="high_pass", cutoff=hp_cutoff_smooth, resonance=hp_resonance_smooth)
        output = Mixer(hp * amplitude_smooth, panning_smooth * 0.5 + 0.5, out_channels=2) # pan all channels in a stereo space with the pansig scaled between 0 and 1
        
        self.set_output(output)

        self.id = str(id(self))
        self.create_ui()

        self.debouncer = ParamSliderDebouncer(PARAM_SLIDER_DEBOUNCE_TIME) if self.num_channels == 1 else None

    def set_input_buf(self, name, value, from_slider=False):
        self.params[name]["buffer"].data[:, :] = value
        if not from_slider and self.num_channels == 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.unobserve_all()
            slider_value = value if self.num_channels == 1 else array2str(value)
            self.debouncer.submit(name, lambda: self.update_slider(slider, slider_value))
        elif not from_slider and self.num_channels > 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.value = array2str(value)
    
    def update_slider(self, slider, value):
        slider.unobserve_all()
        slider.value = value
        slider.observe(
            lambda change: self.set_input_buf(
                    change["owner"].tag, 
                    change["new"],
                    from_slider=True
                ), 
                names="value")

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
        return f"Oscillator {self.id}: {self.name}"
    

class FilteredNoise(Synth):
    def __init__(
            self,
            filter_type="band_pass", # can be 'low_pass', 'band_pass', 'high_pass', 'notch', 'peak', 'low_shelf', 'high_shelf'
            order=3,
            cutoff=440,
            resonance=0.5,
            amplitude=0.5,
            panning=0,
            name="FilteredNoise"):
        super().__init__()

        filter_types = ["low_pass", "band_pass", "high_pass", "notch", "peak", "low_shelf", "high_shelf"]
        assert filter_type in filter_types, f"Filter type must be one of {filter_types}"
        self.filter_type = filter_type
        self.order = np.clip(order, 1, 8)

        self.name = name

        self.params = {
            "cutoff": {
                "min": 20,
                "max": 20000,
                "default": 440,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Cutoff",
                "param_name": "cutoff",
                "owner": self
            },
            "resonance": {
                "min": 0,
                "max": 0.999,
                "default": 0.5,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Resonance",
                "param_name": "resonance",
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
            },
        }
        self.cutoff, self.resonance, self.amplitude, self.panning = broadcast_params(cutoff, resonance, amplitude, panning)
        self.num_channels = len(self.cutoff) # at this point all lengths are the same

        self.cutoff_buffer = sf.Buffer(self.num_channels, 1)
        self.resonance_buffer = sf.Buffer(self.num_channels, 1)
        self.amplitude_buffer = sf.Buffer(self.num_channels, 1)
        self.panning_buffer = sf.Buffer(self.num_channels, 1)
        
        self.cutoff_buffer.data[:, :] = np.array(self.cutoff).reshape(self.num_channels, 1)
        self.params["cutoff"]["buffer"] = self.cutoff_buffer
        self.params["cutoff"]["default"] = self.cutoff

        self.resonance_buffer.data[:, :] = np.array(self.resonance).reshape(self.num_channels, 1)
        self.params["resonance"]["buffer"] = self.resonance_buffer
        self.params["resonance"]["default"] = self.resonance
        
        self.amplitude_buffer.data[:, :] = np.array(self.amplitude).reshape(self.num_channels, 1)
        self.params["amplitude"]["buffer"] = self.amplitude_buffer
        self.params["amplitude"]["default"] = self.amplitude

        self.panning_buffer.data[:, :] = np.array(self.panning).reshape(self.num_channels, 1)
        self.params["panning"]["buffer"] = self.panning_buffer
        self.params["panning"]["default"] = self.panning
        
        self.cutoff_value = sf.BufferPlayer(self.cutoff_buffer, loop=True)
        self.params["cutoff"]["buffer_player"] = self.cutoff_value
        self.resonance_value = sf.BufferPlayer(self.resonance_buffer, loop=True)
        self.params["resonance"]["buffer_player"] = self.resonance_value
        self.amplitude_value = sf.BufferPlayer(self.amplitude_buffer, loop=True)
        self.params["amplitude"]["buffer_player"] = self.amplitude_value
        self.panning_value = sf.BufferPlayer(self.panning_buffer, loop=True)
        self.params["panning"]["buffer_player"] = self.panning_value

        # Clip the resonance value to avoid filter instability
        self.resonance_value_clip = sf.Clip(self.resonance_value, 0, 0.999)
        
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        cutoff_smooth = sf.Smooth(self.cutoff_value, mix_val)
        resonance_smooth = sf.Smooth(self.resonance_value_clip, mix_val)
        amplitude_smooth = sf.Smooth(self.amplitude_value, mix_val)
        panning_smooth = sf.Smooth(self.panning_value, mix_val) # still between -1 and 1

        noise = sf.WhiteNoise()

        # First one
        filters = sf.SVFilter(noise, filter_type=self.filter_type, cutoff=cutoff_smooth, resonance=resonance_smooth)
        # The rest
        for i in range(1, self.order):
            filters = sf.SVFilter(filters, filter_type=self.filter_type, cutoff=cutoff_smooth, resonance=resonance_smooth)
        
        # amplitude compensation
        filters_rms = sf.RMS(filters)
        filters_rms_smooth = sf.Smooth(filters_rms, mix_val)
        filters = filters / filters_rms_smooth
        
        out = Mixer(filters * amplitude_smooth, panning_smooth * 0.5 + 0.5, out_channels=2) # pan all channels in a stereo space with the pansig scaled between 0 and 1
        
        self.set_output(out * 0.707 * 0.5)

        self.id = str(id(self))
        self.create_ui()

        self.debouncer = ParamSliderDebouncer(PARAM_SLIDER_DEBOUNCE_TIME) if self.num_channels == 1 else None

    def set_input_buf(self, name, value, from_slider=False):
        self.params[name]["buffer"].data[:, :] = value
        if not from_slider and self.num_channels == 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.unobserve_all()
            slider_value = value if self.num_channels == 1 else array2str(value)
            self.debouncer.submit(name, lambda: self.update_slider(slider, slider_value))
        elif not from_slider and self.num_channels > 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.value = array2str(value)
    
    def update_slider(self, slider, value):
        slider.unobserve_all()
        slider.value = value
        slider.observe(
            lambda change: self.set_input_buf(
                    change["owner"].tag, 
                    change["new"],
                    from_slider=True
                ), 
                names="value")

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
        return f"FilteredNoise {self.id}: {self.name}"
    

class SimpleFM(Synth):
    def __init__(
            self, 
            carrier_frequency=440,
            harmonicity_ratio=1,
            modulation_index=1, 
            amplitude=0.5, 
            panning=0, 
            lp_cutoff=20000,
            lp_resonance=0.5,
            hp_cutoff=20,
            hp_resonance=0.5,
            name="SimpleFM"):
        super().__init__()

        self.name = name

        self.params = {
            "carrier_freq": {
                "min": 20,
                "max": 8000,
                "default": 440,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Carrier Frequency",
                "param_name": "carrier_freq",
                "owner": self
            },
            "harm_ratio": {
                "min": 0,
                "max": 10,
                "default": 1,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Harmonicity Ratio",
                "param_name": "harm_ratio",
                "owner": self
            },
            "mod_index": {
                "min": 0,
                "max": 10,
                "default": 1,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} Modulation Index",
                "param_name": "mod_index",
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
            },
            "lp_cutoff": {
                "min": 20,
                "max": 20000,
                "default": 20000,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} LP Cutoff",
                "param_name": "lp_cutoff",
                "owner": self
            },
            "lp_resonance": {
                "min": 0,
                "max": 0.999,
                "default": 0,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} LP Resonance",
                "param_name": "lp_resonance",
                "owner": self
            },
            "hp_cutoff": {
                "min": 20,
                "max": 20000,
                "default": 20,
                "unit": "Hz",
                "scale": "log",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} HP Cutoff",
                "param_name": "hp_cutoff",
                "owner": self
            },
            "hp_resonance": {
                "min": 0,
                "max": 0.999,
                "default": 0,
                "unit": "",
                "scale": "linear",
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} HP Resonance",
                "param_name": "hp_resonance",
                "owner": self
            },
        }
        self.carrier_freq, self.harm_ratio, self.mod_index, self.amplitude, self.panning, self.lp_cutoff, self.lp_resonance, self.hp_cutoff, self.hp_resonance = broadcast_params(carrier_frequency, harmonicity_ratio, modulation_index, amplitude, panning, lp_cutoff, lp_resonance, hp_cutoff, hp_resonance)
        self.num_channels = len(self.carrier_freq) # at this point all lengths are the same

        self.carrier_freq_buffer = sf.Buffer(self.num_channels, 1)
        self.harm_ratio_buffer = sf.Buffer(self.num_channels, 1)
        self.mod_index_buffer = sf.Buffer(self.num_channels, 1)
        self.amplitude_buffer = sf.Buffer(self.num_channels, 1)
        self.panning_buffer = sf.Buffer(self.num_channels, 1)
        self.lp_cutoff_buffer = sf.Buffer(self.num_channels, 1)
        self.lp_resonance_buffer = sf.Buffer(self.num_channels, 1)
        self.hp_cutoff_buffer = sf.Buffer(self.num_channels, 1)
        self.hp_resonance_buffer = sf.Buffer(self.num_channels, 1)
        
        self.carrier_freq_buffer.data[:, :] = np.array(self.carrier_freq).reshape(self.num_channels, 1)
        self.params["carrier_freq"]["buffer"] = self.carrier_freq_buffer
        self.params["carrier_freq"]["default"] = self.carrier_freq

        self.harm_ratio_buffer.data[:, :] = np.array(self.harm_ratio).reshape(self.num_channels, 1)
        self.params["harm_ratio"]["buffer"] = self.harm_ratio_buffer
        self.params["harm_ratio"]["default"] = self.harm_ratio

        self.mod_index_buffer.data[:, :] = np.array(self.mod_index).reshape(self.num_channels, 1)
        self.params["mod_index"]["buffer"] = self.mod_index_buffer
        self.params["mod_index"]["default"] = self.mod_index
        
        self.amplitude_buffer.data[:, :] = np.array(self.amplitude).reshape(self.num_channels, 1)
        self.params["amplitude"]["buffer"] = self.amplitude_buffer
        self.params["amplitude"]["default"] = self.amplitude

        self.panning_buffer.data[:, :] = np.array(self.panning).reshape(self.num_channels, 1)
        self.params["panning"]["buffer"] = self.panning_buffer
        self.params["panning"]["default"] = self.panning

        self.lp_cutoff_buffer.data[:, :] = np.array(self.lp_cutoff).reshape(self.num_channels, 1)
        self.params["lp_cutoff"]["buffer"] = self.lp_cutoff_buffer
        self.params["lp_cutoff"]["default"] = self.lp_cutoff

        self.lp_resonance_buffer.data[:, :] = np.array(self.lp_resonance).reshape(self.num_channels, 1)
        self.params["lp_resonance"]["buffer"] = self.lp_resonance_buffer
        self.params["lp_resonance"]["default"] = self.lp_resonance

        self.hp_cutoff_buffer.data[:, :] = np.array(self.hp_cutoff).reshape(self.num_channels, 1)
        self.params["hp_cutoff"]["buffer"] = self.hp_cutoff_buffer
        self.params["hp_cutoff"]["default"] = self.hp_cutoff

        self.hp_resonance_buffer.data[:, :] = np.array(self.hp_resonance).reshape(self.num_channels, 1)
        self.params["hp_resonance"]["buffer"] = self.hp_resonance_buffer
        self.params["hp_resonance"]["default"] = self.hp_resonance
        
        self.carrier_freq_value = sf.BufferPlayer(self.carrier_freq_buffer, loop=True)
        self.params["carrier_freq"]["buffer_player"] = self.carrier_freq_value
        self.harm_ratio_value = sf.BufferPlayer(self.harm_ratio_buffer, loop=True)
        self.params["harm_ratio"]["buffer_player"] = self.harm_ratio_value
        self.mod_index_value = sf.BufferPlayer(self.mod_index_buffer, loop=True)
        self.params["mod_index"]["buffer_player"] = self.mod_index_value
        self.amplitude_value = sf.BufferPlayer(self.amplitude_buffer, loop=True)
        self.params["amplitude"]["buffer_player"] = self.amplitude_value
        self.panning_value = sf.BufferPlayer(self.panning_buffer, loop=True)
        self.params["panning"]["buffer_player"] = self.panning_value
        self.lp_cutoff_value = sf.BufferPlayer(self.lp_cutoff_buffer, loop=True)
        self.params["lp_cutoff"]["buffer_player"] = self.lp_cutoff_value
        self.lp_resonance_value = sf.BufferPlayer(self.lp_resonance_buffer, loop=True)
        self.params["lp_resonance"]["buffer_player"] = self.lp_resonance_value
        self.hp_cutoff_value = sf.BufferPlayer(self.hp_cutoff_buffer, loop=True)
        self.params["hp_cutoff"]["buffer_player"] = self.hp_cutoff_value
        self.hp_resonance_value = sf.BufferPlayer(self.hp_resonance_buffer, loop=True)
        self.params["hp_resonance"]["buffer_player"] = self.hp_resonance_value

        # Clip the resonance values to avoid filter instability
        self.lp_resonance_value_clip = sf.Clip(self.lp_resonance_value, 0, 0.999)
        self.hp_resonance_value_clip = sf.Clip(self.hp_resonance_value, 0, 0.999)
        
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        carrier_freq_smooth = sf.Smooth(self.carrier_freq_value, mix_val)
        harm_ratio_smooth = sf.Smooth(self.harm_ratio_value, mix_val)
        mod_index_smooth = sf.Smooth(self.mod_index_value, mix_val)
        amplitude_smooth = sf.Smooth(self.amplitude_value, mix_val)
        panning_smooth = sf.Smooth(self.panning_value, mix_val) # still between -1 and 1
        lp_cutoff_smooth = sf.Smooth(self.lp_cutoff_value, mix_val)
        lp_resonance_smooth = sf.Smooth(self.lp_resonance_value_clip, mix_val)
        hp_cutoff_smooth = sf.Smooth(self.hp_cutoff_value, mix_val)
        hp_resonance_smooth = sf.Smooth(self.hp_resonance_value_clip, mix_val)

        mod_freq = carrier_freq_smooth * harm_ratio_smooth
        mod_amp = mod_freq * mod_index_smooth
        modulator = sf.SineOscillator(mod_freq) * mod_amp
        carrier = sf.SineOscillator(carrier_freq_smooth + modulator)
        lp = sf.SVFilter(carrier, filter_type="low_pass", cutoff=lp_cutoff_smooth, resonance=lp_resonance_smooth)
        hp = sf.SVFilter(lp, filter_type="high_pass", cutoff=hp_cutoff_smooth, resonance=hp_resonance_smooth)
        output = Mixer(hp * amplitude_smooth, panning_smooth * 0.5 + 0.5, out_channels=2) # pan all channels in a stereo space with the pansig scaled between 0 and 1
        
        self.set_output(output)

        self.id = str(id(self))
        self.create_ui()

        self.debouncer = ParamSliderDebouncer(PARAM_SLIDER_DEBOUNCE_TIME) if self.num_channels == 1 else None

    def set_input_buf(self, name, value, from_slider=False):
        self.params[name]["buffer"].data[:, :] = value
        if not from_slider and self.num_channels == 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.unobserve_all()
            slider_value = value if self.num_channels == 1 else array2str(value)
            self.debouncer.submit(name, lambda: self.update_slider(slider, slider_value))
        elif not from_slider and self.num_channels > 1:
            slider = find_widget_by_tag(self.ui, name)
            slider.value = array2str(value)
    
    def update_slider(self, slider, value):
        slider.unobserve_all()
        slider.value = value
        slider.observe(
            lambda change: self.set_input_buf(
                    change["owner"].tag, 
                    change["new"],
                    from_slider=True
                ), 
                names="value")

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
        return f"Oscillator {self.id}: {self.name}"
    

class Envelope(sf.Patch):
    def __init__(self, attack=0.01, decay=0.01, sustain=0.5, release=0.1, name="Envelope"):
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
        self.name = name
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
        self._ui = EnvelopeCard(self.name, self.id, self.params)
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