"""consecFlagRemoval – remove spike flags that are not sufficiently consecutive."""

import numpy as np


def consec_flag_removal(flag: np.ndarray, consec_rows: int) -> np.ndarray:
    """Clear flag runs longer than *consec_rows*.

    Runs of consecutive flags longer than *consec_rows* are treated as
    physical signal rather than spikes and are unflagged. Port of the
    MATLAB ``consecFlagRemoval`` function.

    Parameters
    ----------
    flag : bool ndarray, shape (N, M)
    consec_rows : int
        Maximum run length still treated as a spike.

    Returns
    -------
    flag : bool ndarray, same shape as input (modified in place copy).
    """
    flag = np.array(flag, dtype=bool)
    if consec_rows >= flag.shape[0]:
        return flag

    # Pad with False at start and end
    padded = np.vstack([np.zeros((1, flag.shape[1]), dtype=bool),
                        flag,
                        np.zeros((1, flag.shape[1]), dtype=bool)])

    flag_flag = padded.copy()
    for i in range(1, consec_rows + 1):
        flag_flag = flag_flag & np.vstack([padded[i:], padded[:i]])

    # Remove padding
    flag_out = padded[1:-1].copy()
    flag_flag = flag_flag[1:-1]

    # Expand consecutive marks forward
    for i in range(1, consec_rows + 1):
        flag_flag = flag_flag | np.vstack([flag_flag[-1:], flag_flag[:-1]])

    flag_out[flag_flag] = False
    return flag_out
