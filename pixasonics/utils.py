import numpy as np
from numba import jit
import threading
import time
from typing import List, Dict

def mix2samps(mixval, eps=1e-6):
    "Convert a mix value (used in sf.Smooth) to samples"
    return np.ceil(np.log(eps) / np.log(mixval))

def samps2mix(samps, eps=1e-6):
    "Convert samples to a mix value (used in sf.Smooth)"
    return eps ** (1 / samps)

def frame2sec(frame, fps):
    "Convert a frame number to seconds"
    return frame / fps

def sec2frame(sec, fps):
    "Convert seconds to a frame number"
    return int(round(sec * fps))

def array2str(arr, decimals=3):
    """String from an array, where elements are rounded to decimals, and the square brackets are removed."""
    return str(np.round(arr, decimals)).replace('[', '').replace(']', '')

@jit(nopython=True)
def scale_array_exp(
    x: np.ndarray,
    in_low: np.ndarray,
    in_high: np.ndarray,
    out_low: np.ndarray,
    out_high: np.ndarray,
    exp: float = 1.0,
) -> np.ndarray:
    """
    Scales an array of values from one range to another. Based on the Max/MSP scale~ object.

    Args:
        x (np.ndarray): The array of values to scale.
        in_low (np.ndarray): The lower bound of the input range.
        in_high (np.ndarray): The upper bound of the input range.
        out_low (np.ndarray): The lower bound of the output range.
        out_high (np.ndarray): The upper bound of the output range.
        exp (float, optional): The exponent to use for the scaling. Defaults to 1.0.

    Returns:
        np.ndarray: The scaled array.
    """
    if np.array_equal(in_high, in_low):
        return np.ones_like(x, dtype=np.float64) * out_high
    else:
        return np.where(
            (x-in_low)/(in_high-in_low) == 0,
            out_low,
            np.where(
                (x-in_low)/(in_high-in_low) > 0,
                out_low + (out_high-out_low) *
                ((x-in_low)/(in_high-in_low))**exp,
                out_low + (out_high-out_low) * -
                ((((-x+in_low)/(in_high-in_low)))**(exp))
            )
        )

@jit(nopython=True)
def resize_interp(
    input: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Resize an array. Uses linear interpolation. Assumes single dim.

    Args:
        input (np.ndarray): Array to resize.
        size (int): The new size of the array.

    Returns:
        np.ndarray: The resized array.
    """
    # create x axis for input
    input_x = np.arange(0, len(input))
    # create array with sampling indices
    output_x = np.linspace(0, len(input_x)-1, size)
    # interpolate
    return np.interp(output_x, input_x, input)#.astype(np.float64)


def filter_matrix(
        matrix, 
        filter_rows=None,
        filter_cols=None,
        filter_chans=None,
        filter_layers=None):
    """Filter a 4D matrix based on the provided filter inputs. 
    The filter inputs can be None, int, slice, list, or str (e.g. "0:3")."""
    filter_inputs = [filter_rows, filter_cols, filter_chans, filter_layers]
    filter_slices = [None] * 4

    for i, filter_input in enumerate(filter_inputs):
        if filter_input is None:
            filter_slices[i] = slice(None)
        elif isinstance(filter_input, int):
            filter_slices[i] = slice(filter_input, filter_input + 1)
        elif isinstance(filter_input, slice):
            filter_slices[i] = filter_input
        elif isinstance(filter_input, list):
            filter_slices[i] = filter_input
        elif isinstance(filter_input, str): # e.g. "0:3"
            filter_slices[i] = slice(*map(int, filter_input.split(":")))
        else:
            raise ValueError(f"Invalid filter input: {filter_input}")

    return matrix[filter_slices[0], filter_slices[1], filter_slices[2], filter_slices[3]]

def test_filter_matrix():
    # tests
    a = np.random.rand(100, 100, 20, 10)
    b = filter_matrix(a, None, None, None, None)
    assert b.shape == a.shape
    b = filter_matrix(a, 0, None, None, None)
    assert b.shape == (1, 100, 20, 10)
    b = filter_matrix(a, slice(0, 3), None, None, None) 
    assert b.shape == (3, 100, 20, 10)
    b = filter_matrix(a, [0, 2, 5], None, None, None)
    assert b.shape == (3, 100, 20, 10)
    b = filter_matrix(a, "0:3", None, None, None)
    assert b.shape == (3, 100, 20, 10)
    b = filter_matrix(a, None, None, None, 0)
    assert b.shape == (100, 100, 20, 1)
    b = filter_matrix(a, None, None, None, slice(0, 3))
    assert b.shape == (100, 100, 20, 3)
    b = filter_matrix(a, None, None, None, [0, 2, 5])
    assert b.shape == (100, 100, 20, 3)
    b = filter_matrix(a, None, None, None, "0:3")
    assert b.shape == (100, 100, 20, 3)
    b = filter_matrix(a, 0, 0, 0, 0)
    assert b.shape == (1, 1, 1, 1)
    b = filter_matrix(a, 0, 0, 0, slice(0, 3))
    assert b.shape == (1, 1, 1, 3)
    b = filter_matrix(a, 0, 0, 0, [0, 2, 5])
    assert b.shape == (1, 1, 1, 3)
    b = filter_matrix(a, 0, 0, 0, "0:3")
    assert b.shape == (1, 1, 1, 3)


def broadcast_params(*param_lists):
    """Helper function to broadcast and interpolate all param lists to the same length."""
    # if an input is a numpy array, convert it to a list
    param_lists = [p.tolist() if isinstance(p, np.ndarray) else p for p in param_lists]
    # if an input list is just a single value, convert it to a list
    param_lists = [p if isinstance(p, list) else [p] for p in param_lists]
    max_len = max([len(p) for p in param_lists])
    broadcasted_params = []
    for plist in param_lists:
        # convert values to floats
        plist = [float(p) for p in plist]
        if len(plist) < max_len:
            # interpolate
            plist = resize_interp(plist, max_len).tolist()
        broadcasted_params.append(plist)
    return broadcasted_params


def find_dict_with_entry(list_of_dicts: List[Dict], key: str, value) -> Dict:
    """
    Find a dictionary in a list where a specific key has a given value.
    
    Args:
        list_of_dicts (list): A list of dictionaries to search through
        key (str): The key to search for in each dictionary
        value: The value to match against the key
        
    Returns:
        dict: The first dictionary where dict[key] == value, or None if no match is found
    """
    return next((d for d in list_of_dicts if key in d and d[key] == value), None)


class Timer:
    def __init__(self, timeout, callback, manager):
        self._timeout = timeout
        self._callback = callback
        self._manager = manager
        self._start_time = None
        self._cancel_event = threading.Event()

    def start(self):
        self._start_time = time.time()
        self._manager.add_timer(self)

    def cancel(self):
        self._cancel_event.set()
        self._manager.remove_timer(self)

    @property
    def scheduled(self):
        return self in self._manager.timers

    def remaining_time(self):
        if self._start_time is None:
            return float('inf')
        elapsed = time.time() - self._start_time
        return max(0, self._timeout - elapsed)

    def execute(self):
        if not self._cancel_event.is_set() and self._callback is not None:
            self._callback()

class TimerManager:
    def __init__(self):
        self.timers = []
        self.lock = threading.Lock()
        self.new_timer_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add_timer(self, timer):
        with self.lock:
            self.timers.append(timer)
            self.new_timer_event.set()

    def remove_timer(self, timer):
        with self.lock:
            if timer in self.timers:
                self.timers.remove(timer)
                self.new_timer_event.set()

    def _run(self):
        while True:
            with self.lock:
                if not self.timers:
                    self.new_timer_event.clear()

            self.new_timer_event.wait()

            with self.lock:
                self.timers.sort(key=lambda t: t.remaining_time())
                for timer in self.timers:
                    if timer.remaining_time() <= 0:
                        timer.execute()
                        self.timers.remove(timer)

            time.sleep(0.01)

class ParamSliderDebouncer:
    def __init__(self, wait=0.5):
        self.wait = wait
        self.manager = TimerManager()
        self.timers = {}

    def submit(self, key, callback):
        if key in self.timers:
            self.timers[key].cancel()
        self.timers[key] = Timer(self.wait, callback, self.manager)
        self.timers[key].start()

class ParamSliderThrottler:
    def __init__(self, wait=0.5):
        self.wait = wait
        self.manager = TimerManager()
        self.timers = {}

    def submit(self, key, callback):
        if key not in self.timers:
            self.timers[key] = Timer(self.wait, callback, self.manager)
            self.timers[key].start()
        else:
            timer = self.timers[key]
            if not timer.scheduled:
                timer = Timer(self.wait, callback, self.manager)
                timer.start()