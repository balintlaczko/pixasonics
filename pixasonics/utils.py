import numpy as np
from numba import jit

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
    # if an input list is just a single value, convert it to a list
    param_lists = [p if isinstance(p, list) else [p] for p in param_lists]
    max_len = max([len(p) for p in param_lists])
    broadcasted_params = []
    for plist in param_lists:
        if len(plist) < max_len:
            # interpolate
            plist = resize_interp(plist, max_len).tolist()
        broadcasted_params.append(plist)
    return broadcasted_params

