""" Scripts to perform linear regression methods
Based on Hastie/Tibshirani/Friedman "Elements of Statistical Learning"
Ch 3 - Linear Methods for Regression

Author: John Lee
Oct 2024
"""

import numpy as np
from typing import Union
from scipy.linalg import svd

from learn_htf.core.matrix import Matrix
from learn_htf.core.model import Model
from learn_htf.utils.metrics import lp_norm
# TODO Principle component regression
# TODO SVD/cholesky/QR decomp
# TODO handle sparse data

# follow an sklearn approach of .fit and .predict
# model.fit()


class LinearRegression(Model):
    """
    Linear regression model using least squares
    """

    def __init__(self, ridge_lambda: float = 0):
        super().__init__(ridge_lambda=ridge_lambda)
        self.ridge_lambda = self.model_params['ridge_lambda']

    def _fit(self, x: Union[Matrix, np.ndarray], y: Union[Matrix, np.ndarray]):
        """
        Calculates the coefficients (B_i) of a linear fit
            f(X) = B_0 + Sum( X_i B_i )
        that minimizes ||f(X) - y||


        Args:
            x (np.ndarray): Size N x P
            y (np.ndarray): Size N x K

        Returns:
            _type_: Coefficients of linear fit: Size M+1 x K
        """
        if isinstance(x, np.ndarray):
            x = Matrix(x)
        if isinstance(y, np.ndarray):
            y = Matrix(y)
        xfeats = x.features

        assert x.shape[0] == y.shape[0]

        n, p = x.shape
        x = Matrix(np.hstack([Matrix(np.ones(n)), x]))
        coef = np.linalg.solve(
            x.T @ x - self.model_params['ridge_lambda'] * Matrix(np.eye(p+1)), x.T @ y)

        samps = [fr'beta_{i}' for i in range(p+1)]
        # TODO use feature space instead of betas - need to include the intercept as well
        coef = Matrix(coef, samples=samps, features=xfeats)

        return coef

    def _predict(self, x):
        """ TODO
        """
        xsamps = x.samples
        xfeats = x.features
        n = x.shape[0]

        x = Matrix(np.hstack([Matrix(np.ones(n)), x]))

        preds = Matrix((self.coefficients.T @ x.T).T)
        preds.samples = xsamps
        preds.features = xfeats

        return preds

    def score(self,  y: Matrix, predictions: Matrix):
        """ TODO
        """

        return lp_norm(y, predictions, p=2)


class PrincipleComponentsRegression(Model):

    def _fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def score(self, x, y):
        pass


def principle_components_regression(x: np.ndarray, y: np.ndarray, pct_variance=.9, x_centered=False):

    b0 = y.mean(axis=0)

    if not x_centered:
        x = x - x.mean(axis=0)
    u, d, vt = svd(x, full_matrices=False)

    # get the first PC where we reach pct_variance
    explanined_variance = d**2 / d.sum(0)
    npc = np.nonzero(np.cumsum(explanined_variance) > pct_variance)[0][0]

    # x@vt.T
    # (y - b0 + u[:, :npc]@u[:, :npc].T@y)
