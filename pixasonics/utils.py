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
    if in_high == in_low:
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