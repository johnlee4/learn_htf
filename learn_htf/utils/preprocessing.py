import numpy as np
from learn_htf.core.matrix import Matrix

# scaling normalization etc
# TODO  Do we need a fit / transform /inverse transform approach?
# TODO min/max scale
# log transform
# power transform (box-cox)
# polynomial transform


def z_scale(x: Matrix, axis=0):
    """

    Args:
        x (Matrix): _description_

    Returns:
        _type_: _description_
    """
    mns = x.mean(axis=axis, keepdims=True)
    stds = x.std(axis=axis, keepdims=True)

    return (x - mns)/stds, mns, stds


def expand_basis(x: Matrix, funcs=None, include_input=True):
    """ Basis Expansion to allow to arbitrary function fitting

    Args:
        x (Matrix): _description_
        funcs (list, optional): _description_. Defaults to [].
        include_input (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if isinstance(funcs, list):
        for func in funcs:
            if not callable(func):
                raise TypeError(f'{func} is not callable!')

    if funcs is None:
        funcs = []

    if callable(funcs):
        funcs = [funcs]

        # store the un-adultered input
    xfeats = x.features.index
    if include_input:
        results = [x]
        features = [f'{feat}'for feat in xfeats]
    else:
        results = []
        features = []
    for func in funcs:
        results.append(func(x))
        features += [f'{func.__name__}({feat})' for feat in xfeats]

    return Matrix(np.hstack(results), features=features, samples=x.samples)
