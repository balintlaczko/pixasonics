import numpy as np
import signalflow as sf
import json
from .utils import broadcast_params, array2str, find_dict_with_entry, ParamSliderDebouncer
from .ui import SynthCard, EnvelopeCard, find_widget_by_tag
from typing import Dict, List
import copy

PARAM_SLIDER_DEBOUNCE_TIME = 0.05

class Synth():
    def __init__(
            self, 
            patch_spec: sf.PatchSpec, 
            params_dict: Dict = None, 
            name: str = "Synth",
            add_amplitude: bool = True,
            add_panning: bool = True):
        self.name = name
        self.patch_spec = patch_spec
        self.params = {} if params_dict is None else copy.deepcopy(params_dict)
        self.synth = None
        self.num_channels = -1 # will be set in generate_params
        self.add_amplitude = add_amplitude
        self.add_panning = add_panning
        self.id = str(id(self))
        # generate params dict
        self.generate_params()
        # create param buffers, their players, smoothing, and the patch
        self.create_audio_graph()
        # create ui
        self.create_ui()
        # create param slider debouncer if necessary
        self.debouncer = ParamSliderDebouncer(PARAM_SLIDER_DEBOUNCE_TIME) if self.num_channels == 1 else None

    def generate_params(self):
        # check that there are keys for all params
        params = get_spec_inputs_dict(self.patch_spec)
        # add amplitude if requested
        if self.add_amplitude:
            # amplitude should default to 0.5
            params["amplitude"] = 0.5
            self.params["amplitude"] = {
                "min": 0,
                "max": 1,
            }
        # add panning if requested
        if self.add_panning:
            # panning should default to middle if single channel or a spread if multichannel
            spec_output_channels = get_patch_spec_num_output_channels(self.patch_spec)
            if spec_output_channels == 1:
                panning_default = 0
            else:
                panning_default = [-1, 1]
            params["panning"] = panning_default
            self.params["panning"] = {
                "min": -1,
                "max": 1,
            }
        # parse & sanitize param dicts
        for param_name in params.keys():
            # if not, then make a new dict for it
            if param_name not in self.params:
                self.params[param_name] = {}
            # if yes, then assert that it is a dict
            assert isinstance(self.params[param_name], dict)
            # fill in the dict with the values
            template = {
                "min": 0,
                "max": 1,
                "unit": "",
                "scale": "linear",
            }
            forced_template = {
                "buffer": None,
                "buffer_player": None,
                "name": f"{self.name} {param_name.capitalize()}",
                "param_name": param_name,
                "owner": self,
            }
            # combine template with user-provided values
            self.params[param_name] = {**template, **self.params[param_name]} # user can overwrite the template
            # combine forced_template with user-provided values
            self.params[param_name] = {**self.params[param_name], **forced_template} # user cannot overwrite the forced_template
        # broadcast params
        param_names = list(params.keys())
        param_values = [params[param_name] for param_name in param_names]
        broadcasted_params = broadcast_params(*param_values)
        self.num_channels = len(broadcasted_params[0]) # all params should have the same amount of channels now
        # register values in params dict
        for i, param_name in enumerate(param_names):
            params_list = broadcasted_params[i]
            self.params[param_name]["default"] = params_list
            self.params[param_name]["value"] = params_list
    
    def create_audio_graph(self):
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        self.patch = sf.Patch(self.patch_spec)
        # generate param buffers & players & smoothers for each param
        param_names = list(self.params.keys())
        for param_name in param_names:
            params_list = self.params[param_name]["value"] # expecting the broadcasted values
            buffer = sf.Buffer(self.num_channels, 1)
            buffer.data[:, :] = np.array(params_list).reshape(self.num_channels, 1)
            self.params[param_name]["buffer"] = buffer
            buffer_player = sf.BufferPlayer(buffer, loop=True)
            self.params[param_name]["buffer_player"] = buffer_player
            smoothed = sf.Smooth(buffer_player, mix_val)
            self.params[param_name]["smoothed"] = smoothed
            # set the input of the patch to the smoothed value, except the added amplitude and panning params
            if self.add_amplitude and param_name == "amplitude":
                continue
            if self.add_panning and param_name == "panning":
                continue
            self.patch.set_input(param_name, smoothed)
        # multiply patch output with amplitude if requested
        self.patch_output = self.patch.output * self.params["amplitude"]["smoothed"] if self.add_amplitude else self.patch.output
        # use Mixer to mix down to stereo if panning is requested
        self.output = Mixer(self.patch_output, self.params["panning"]["smoothed"] * 0.5 + 0.5, out_channels=2) if self.add_panning else self.patch_output

    def set_input_buf(self, name, value, from_slider=False):
        self.params[name]["buffer"].data[:, :] = value
        self.params[name]["value"] = value.tolist() if isinstance(value, np.ndarray) else value
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
        return f"Synth {self.id}: {self.name}"


class ThereminPatch(sf.Patch):
    def __init__(self, frequency=440):
        super().__init__()
        frequency = self.add_input("frequency", frequency)
        out = sf.SineOscillator(frequency)
        self.set_output(out)


class Theremin(Synth):
    def __init__(self, frequency=440, name="Theremin"):
        # create the patch spec
        _spec = ThereminPatch(
            frequency=frequency, 
        ).to_spec()
        # create the params dict
        _params = {
            "frequency": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
        }
        # call the parent constructor
        super().__init__(_spec, params_dict=_params, name=name)
        
    def __repr__(self):
        return f"Theremin {self.id}: {self.name}"


class OscillatorPatch(sf.Patch):
    def __init__(
            self, 
            frequency=440, 
            lp_cutoff=20000,
            lp_resonance=0.5,
            hp_cutoff=20,
            hp_resonance=0.5,
            waveform="sine",
            ):
        super().__init__()
        wf_types = ["sine", "square", "saw", "triangle"]
        assert waveform in wf_types, f"Waveform must be one of {wf_types}"
        frequency = self.add_input("frequency", frequency)
        lp_cutoff = self.add_input("lp_cutoff", lp_cutoff)
        lp_resonance = self.add_input("lp_resonance", lp_resonance)
        hp_cutoff = self.add_input("hp_cutoff", hp_cutoff)
        hp_resonance = self.add_input("hp_resonance", hp_resonance)
        # create the synth
        osc_templates = [sf.SineOscillator, sf.SquareOscillator, sf.SawOscillator, sf.TriangleOscillator]
        osc = osc_templates[wf_types.index(waveform)](frequency)
        # create the filters
        lp_resonance_clipped = sf.Clip(lp_resonance, 0.0, 0.999)
        hp_resonance_clipped = sf.Clip(hp_resonance, 0.0, 0.999)
        lp = sf.SVFilter(
            osc,
            filter_type="low_pass",
            cutoff=lp_cutoff,
            resonance=lp_resonance_clipped
        )
        hp = sf.SVFilter(
            lp,
            filter_type="high_pass",
            cutoff=hp_cutoff,
            resonance=hp_resonance_clipped
        )
        out = hp
        self.set_output(out)


class Oscillator(Synth):
    def __init__(
        self,
        frequency=440, 
        lp_cutoff=20000,
        lp_resonance=0.5,
        hp_cutoff=20,
        hp_resonance=0.5,
        waveform="sine",
        name="Oscillator",
        ):
        # create the patch spec
        _spec = OscillatorPatch(
            frequency=frequency, 
            lp_cutoff=lp_cutoff,
            lp_resonance=lp_resonance,
            hp_cutoff=hp_cutoff,
            hp_resonance=hp_resonance,
            waveform=waveform
        ).to_spec()
        # create the params dict
        _params = {
            "frequency": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "lp_cutoff": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "lp_resonance": {
                "min": 0,
                "max": 0.999,
            },
            "hp_cutoff": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "hp_resonance": {
                "min": 0,
                "max": 0.999,
            },
        }
        # call the parent constructor
        super().__init__(_spec, params_dict=_params, name=name)

    def __repr__(self):
        return f"Oscillator {self.id}: {self.name}"


class FilteredNoisePatch(sf.Patch):
    def __init__(
            self,
            filter_type="band_pass", # can be 'low_pass', 'band_pass', 'high_pass', 'notch', 'peak', 'low_shelf', 'high_shelf'
            order=3,
            cutoff=440,
            resonance=0.5,
            ):
        super().__init__()
        filter_types = ["low_pass", "band_pass", "high_pass", "notch", "peak", "low_shelf", "high_shelf"]
        assert filter_type in filter_types, f"Filter type must be one of {filter_types}"
        self.filter_type = filter_type
        self.order = np.clip(order, 1, 8)
        cutoff = self.add_input("cutoff", cutoff)
        resonance = self.add_input("resonance", resonance)
        resonance_clipped = sf.Clip(resonance, 0.0, 0.999)
        graph = sf.AudioGraph.get_shared_graph()
        mix_val = sf.calculate_decay_coefficient(0.05, graph.sample_rate, 0.001)
        # create the synth
        noise = sf.WhiteNoise()
        # first one
        filters = sf.SVFilter(
            noise,
            filter_type=self.filter_type,
            cutoff=cutoff,
            resonance=resonance_clipped
        )
        # the rest
        for i in range(1, self.order):
            filters = sf.SVFilter(
                filters,
                filter_type=self.filter_type,
                cutoff=cutoff,
                resonance=resonance_clipped
            )
        # amplitude compensation
        filters_rms = sf.RMS(filters)
        filters_rms_smooth = sf.Smooth(filters_rms, mix_val)
        filters = filters / filters_rms_smooth
        # output
        out = filters
        self.set_output(out * 0.707 * 0.5)


class FilteredNoise(Synth):
    def __init__(
            self,
            filter_type="band_pass", # can be 'low_pass', 'band_pass', 'high_pass', 'notch', 'peak', 'low_shelf', 'high_shelf'
            order=3,
            cutoff=440,
            resonance=0.5,
            name="FilteredNoise",
            ):
        # create the patch spec
        _spec = FilteredNoisePatch(
            filter_type=filter_type,
            order=order,
            cutoff=cutoff,
            resonance=resonance,
        ).to_spec()
        # create the params dict
        _params = {
            "cutoff": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "resonance": {
                "min": 0,
                "max": 0.999,
            },
        }
        # call the parent constructor
        super().__init__(_spec, params_dict=_params, name=name)

    def __repr__(self):
        return f"FilteredNoise {self.id}: {self.name}"


class SimpleFMPatch(sf.Patch):
    def __init__(
            self, 
            carrier_frequency=440, 
            harmonicity_ratio=1,
            modulation_index=1,
            lp_cutoff=20000,
            lp_resonance=0.5,
            hp_cutoff=20,
            hp_resonance=0.5,
            ):
        super().__init__()
        carrier_freq = self.add_input("carrier_freq", carrier_frequency)
        harm_ratio = self.add_input("harm_ratio", harmonicity_ratio)
        mod_index = self.add_input("mod_index", modulation_index)
        lp_cutoff = self.add_input("lp_cutoff", lp_cutoff)
        lp_resonance = self.add_input("lp_resonance", lp_resonance)
        hp_cutoff = self.add_input("hp_cutoff", hp_cutoff)
        hp_resonance = self.add_input("hp_resonance", hp_resonance)
        # create the synth
        mod_freq = carrier_freq * harm_ratio
        mod_amp = mod_freq * mod_index
        modulator = sf.SineOscillator(mod_freq) * mod_amp
        carrier = sf.SineOscillator(carrier_freq + modulator)
        # create the filters
        lp_resonance_clipped = sf.Clip(lp_resonance, 0.0, 0.999)
        hp_resonance_clipped = sf.Clip(hp_resonance, 0.0, 0.999)
        lp = sf.SVFilter(
            carrier,
            filter_type="low_pass",
            cutoff=lp_cutoff,
            resonance=lp_resonance_clipped
        )
        hp = sf.SVFilter(
            lp,
            filter_type="high_pass",
            cutoff=hp_cutoff,
            resonance=hp_resonance_clipped
        )
        out = hp
        self.set_output(out)


class SimpleFM(Synth):
    def __init__(
        self,
        carrier_frequency=440, 
        harmonicity_ratio=1,
        modulation_index=1,
        lp_cutoff=20000,
        lp_resonance=0.5,
        hp_cutoff=20,
        hp_resonance=0.5,
        name="SimpleFM",
    ):
        # create the patch spec
        _spec = SimpleFMPatch(
            carrier_frequency=carrier_frequency, 
            harmonicity_ratio=harmonicity_ratio,
            modulation_index=modulation_index,
            lp_cutoff=lp_cutoff,
            lp_resonance=lp_resonance,
            hp_cutoff=hp_cutoff,
            hp_resonance=hp_resonance
        ).to_spec()
        # create the params dict
        _params = {
            "carrier_freq": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "harm_ratio": {
                "min": 0,
                "max": 10,
            },
            "mod_index": {
                "min": 0,
                "max": 10,
            },
            "lp_cutoff": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "lp_resonance": {
                "min": 0,
                "max": 0.999,
            },
            "hp_cutoff": {
                "min": 20,
                "max": 20000,
                "unit": "Hz",
                "scale": "log",
            },
            "hp_resonance": {
                "min": 0,
                "max": 0.999,
            },
        }
        # call the parent constructor
        super().__init__(_spec, params_dict=_params, name=name)
    
    def __repr__(self):
        return f"SimpleFM {self.id}: {self.name}"
    

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
        self.params["attack"]["value"] = attack
        self.params["decay"]["default"] = decay
        self.params["decay"]["value"] = decay
        self.params["sustain"]["default"] = sustain
        self.params["sustain"]["value"] = sustain
        self.params["release"]["default"] = release
        self.params["release"]["value"] = release

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


def patch_spec2dict(spec: sf.PatchSpec) -> Dict:
    """
    Convert a patch spec to a python dict.
    """
    spec_dict = json.loads(spec.to_json())
    return spec_dict


def patch_dict2spec(spec_dict: Dict) -> sf.PatchSpec:
    """
    Convert a python dict to a patch spec.
    """
    spec = sf.PatchSpec.from_json(json.dumps(spec_dict))
    return spec


def get_spec_input_names(spec: sf.PatchSpec) -> List[str]:
    """
    Get the inputs of a patch spec.
    """
    spec_dict = patch_spec2dict(spec)
    inputs = spec_dict["inputs"]
    input_names = [input["patch_input_name"] for input in inputs]
    return input_names


def get_spec_inputs_dict(spec: sf.PatchSpec) -> Dict:
    """
    Get the inputs of a patch spec as a dict, where keys are the
    'patch_input_name' and values are the corresponding node values.
    """
    spec_dict = patch_spec2dict(spec)
    inputs = spec_dict["inputs"]
    nodes = spec_dict["nodes"]
    out_dict = {}
    for input in inputs:
        input_name = input["patch_input_name"]
        node_id = input["node_id"]
        node_input_name = input["node_input_name"]
        # find the node with the same id
        node = find_dict_with_entry(nodes, "id", node_id)
        input_value = node["inputs"][node_input_name]
        # if the value is a dict it means there is a channel-array node that holds the values
        if isinstance(input_value, dict):
            channel_array_id = input_value["id"]
            # find the channel array node
            channel_array_node = find_dict_with_entry(nodes, "id", channel_array_id)
            # check that it is really a channel array node
            assert channel_array_node["node"] == "channel-array"
            # get the values
            input_value = channel_array_node["inputs"] # this is a dict
            # convert to list
            input_value = [val for val in input_value.values()]
        out_dict[input_name] = input_value
    return out_dict


def get_patch_spec_num_output_channels(spec: sf.PatchSpec) -> int:
    """
    Get the number of output channels of a patch spec.
    """
    patch = sf.Patch(spec)
    return patch.output.num_output_channels
