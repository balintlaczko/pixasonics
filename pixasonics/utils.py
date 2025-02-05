import numpy as np

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

# @jit(nopython=True)
# TODO: add numba support?
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
    return np.interp(output_x, input_x, input)


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