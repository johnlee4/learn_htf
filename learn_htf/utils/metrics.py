"""
list of metrics
"""
from learn_htf.core.matrix import Matrix


def lp_norm(y1: Matrix, y2: Matrix, p: float):
    """_summary_

    Args:
        y1 (Matrix): _description_
        y2 (Matrix): _description_
        p (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
    if p < 0:
        raise ValueError(f'p should be >= 0. Got {p} ')

    if p == 0:
        pass
    elif p < 1:
        pass
    elif p >= 1:
        pass

    norm = 1
    return norm
