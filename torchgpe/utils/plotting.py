import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np


def pi_tick_formatter(val, _):
    """Formats a tick value in multiples of pi.

    Args:
        val (float): The tick value.
        _ (int): The tick position.
    """
    if val == 0:
        return 0
    if (val/np.pi*2) % 2 == 0:
        return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi)==1 else int(val/np.pi)}\\pi$"
    return f"${('+' if np.sign(val)==1 else '-') if abs(val/np.pi*2)==1 else int(val/np.pi*2)}\\pi / 2$"

