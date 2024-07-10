import numpy as np

def sig_stars(p):
    """
    Returns number of significant asterices from p value.
    Args:
        p (float): p value
    Returns:
        number of asterices (int)
    """
    if p <= .01:
        return int(-np.ceil(np.log10(abs(p)))) # should be correct needs checking
    elif p <= .05:
        return 1
    else:
        return 0